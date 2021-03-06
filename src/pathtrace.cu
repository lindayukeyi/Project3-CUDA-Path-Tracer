#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>
#include <cuda_runtime.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include <chrono>

#define ERRORCHECK 1
#define STREAMCOMPACTION 1
#define SORTBYMATERIAL 0
#define CACHE 1
#define DOF 0
#define OCTREE 0
#define ANTIALISING 0
#define BLUR 0
#define CULLING 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static PathSegment* dev_paths_cache = NULL;
static ShadeableIntersection* dev_intersections_cache = NULL;
static Triangle* dev_triangles = NULL;
static OctreeNode* dev_octrees = NULL;



void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_paths_cache, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections_cache, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_octrees, scene->octrees.size() * sizeof(OctreeNode));
    cudaMemcpy(dev_octrees, scene->octrees.data(), scene->octrees.size() * sizeof(OctreeNode), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_triangles);
    cudaFree(dev_octrees);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, float lensRaduis, float focalDistance, glm::vec3 speed)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;



    if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
        thrust::uniform_real_distribution<float> u01(0, 1);
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);

		PathSegment & segment = pathSegments[index];
        glm::vec3 new_camera_view = cam.view;

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

#if BLUR
        thrust::normal_distribution<float> n01(0, 1);
        float t = abs(n01(rng));
        new_camera_view = cam.view * (1 - t) + (speed + cam.view) * t;
#endif // BLUR

		// TODO: implement antialiasing by jittering the ray
#if ANTIALISING
    segment.ray.direction = glm::normalize(new_camera_view
        - cam.right * cam.pixelLength.x * ((float)x + u01(rng)  - (float)cam.resolution.x * 0.5f)
        - cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
    );
#else
		segment.ray.direction = glm::normalize(new_camera_view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);
#endif
#if DOF
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        glm::vec2 lensPointSph(u01(rng) * 2 * PI, u01(rng) * lensRaduis);
        glm::vec2 lensPoint(lensPointSph.y * glm::cos(lensPointSph.x), lensPointSph.y * glm::sin(lensPointSph.x));
        float ft = focalDistance;
        glm::vec3 pFocus = getPointOnRay(segment.ray, ft);

        segment.ray.origin = cam.position + glm::vec3(lensPoint.x, lensPoint.y, 0);

        segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
#endif // DOF       
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections,
    Triangle* triangles,
    OctreeNode* octrees,
    int numOctreeNodes
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
        glm::vec2 uv;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?
            else if (geom.type == MESH) {
#if CULLING
                // Check bbox
                t = meshBboxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);

                if (t != -1) {
                    // for (int i = geom.startTriangleIndex; i <= geom.endTriangleIndex; ++i) {
                        // Triangle &tri = triangles[2];
#endif
#if OCTREE

                    t = octreeIntersectionTest(geom, triangles, octrees, pathSegment.ray, tmp_intersect, tmp_normal, outside, numOctreeNodes);
#else
                    t = triangleIntersectionTest(geom, triangles, pathSegment.ray, tmp_intersect, tmp_normal, outside, tmp_uv);
#endif

                    //}
#if CULLING
                }
#endif // CULLING


                
            }
			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
                uv = tmp_uv;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = uv;
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial (
  int iter
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials,
    int depth,
    int traceDepth
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
        pathSegments[idx].remainingBounces = -1;
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        //float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        //pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        //pathSegments[idx].color *= u01(rng); // apply some noise because why not
          
          if (depth == traceDepth || pathSegments[idx].remainingBounces == 0) {
              pathSegments[idx].color = glm::vec3(0.0f);
              pathSegments[idx].remainingBounces--;
          }
          else {
              glm::vec3 intesect = getPointOnRay(pathSegments[idx].ray, intersection.t);
              glm::vec3 normal = intersection.surfaceNormal;
              scatterRay(pathSegments[idx], intesect, normal, material, rng, intersection.uv, intersection.materialId);
              pathSegments[idx].remainingBounces--;
          }

      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
      pathSegments[idx].remainingBounces = -1;
    }
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

// thrust::remove_if helper function
struct is_terminate
{
    __host__ __device__
        bool operator()(const PathSegment &path) {
        return path.remainingBounces > 0;
    }
};

// Comparator
struct myCompare
{
    __host__ __device__
        bool operator()(const ShadeableIntersection &m1, const ShadeableIntersection &m2) {
        return m1.materialId < m2.materialId;
    }

};


__global__ void kernShuffle(int num_paths, int* ray_indices, PathSegment* paths, PathSegment* paths_shuffle) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index >= num_paths) {
        return;
    }

    paths_shuffle[index] = paths[ray_indices[index]];
}
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    int num_octreeNodes = hst_scene->octrees.size();
    float lensRadius = 0.01;
    float focalDistance = 8;
    glm::vec3 speed(0.1f, 0.0f, -0.0f);

    printf("\n\niter: %d   \n", iter);

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing


    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, lensRadius, focalDistance, speed);
    checkCUDAError("generate camera ray");

	int depth = 1;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
    printf("depth: %d  nums_path:  %d\n", depth - 1, num_paths);

  bool iterationComplete = false;
  auto start = std::chrono::steady_clock::now();
	while (!iterationComplete) {

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE
    if (iter == 1 && depth == 1) {
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections
            , dev_triangles
            , dev_octrees
            , num_octreeNodes
            );
        checkCUDAError("octree failure");
        cudaDeviceSynchronize();
        //cudaMemcpy(dev_paths_cache, dev_paths, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_intersections_cache, dev_intersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
    }
    else if (iter != 1 && depth == 1) {
        //cudaMemcpy(dev_paths, dev_paths_cache, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_intersections, dev_intersections_cache, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
    }
    else {
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections
            , dev_triangles
            , dev_octrees
            , num_octreeNodes
            );
        checkCUDAError("octree failure");
        cudaDeviceSynchronize();
    }
#else
    computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
        depth
        , num_paths
        , dev_paths
        , dev_geoms
        , hst_scene->geoms.size()
        , dev_intersections
        , dev_triangles
        , dev_octrees
        , num_octreeNodes
        );
    checkCUDAError("trace one bounce");
    cudaDeviceSynchronize();
#endif // CACHE



	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.
  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.



#if SORTBYMATERIAL
    thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);
    thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(dev_intersections);

    thrust::sort_by_key(dev_thrust_intersections, dev_thrust_intersections + num_paths, dev_thrust_paths, myCompare());

#endif // SORTBYMATERIAL

    shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
        iter,
        num_paths,
        dev_intersections,
        dev_paths,
        dev_materials,
        depth,
        traceDepth
        );
#if STREAMCOMPACTION
    PathSegment *new_end =  thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, is_terminate());
    num_paths = new_end - dev_paths;
#endif
    printf("depth: %d  nums_path:  %d\n", depth, num_paths);
    if (num_paths <= 0) {
        depth = traceDepth;
        //printf("no bounces!\n");
    }
        // tracing

        depth++;
  if (depth > traceDepth) {
      iterationComplete = true; // TODO: should be based off stream compaction results.
  }
	}

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Iter:" << iter << ",   elapsed time: " << elapsed_seconds.count() << "s\n";
    //printf("%d :  %d\n", iter, num_paths);
    num_paths = dev_path_end - dev_paths;
  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
