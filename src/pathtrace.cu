#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>
#include <cuda_runtime.h>
#include <device_functions.h>

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
#define GAUSSIAN 0
#define ATROUS 0

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



float kernel[25] = {
1.0f/256.0f,	1.0f/64.0f,	3.0f/128.0f,	1.0f / 64.0f,	1.0f / 256.0f,
1.0f / 64.0f,	1.0f / 16.0f,	3.0f / 32.0f,	1.0f/16.0f,	1.0f / 64.0f,
3.0f / 128.0f,	3.0f/32.0f,	9.0f/64.0f,	3.0f / 32.0f,	3.0f / 128.0f,
1.0f / 64.0f,	1.0f / 16.0f,	3.0f / 32.0f,	1.0f / 16.0f,	1.0f / 64.0f,
1.0f / 256.0f,	1.0f / 64.0f,	3.0f / 128.0f,	1.0f / 64.0f,	1.0f / 256.0f };
 
/*
float kernel[25] = {
    1.0/16.0, 1.0/4.0, 3.0/8.0, 1.0/4.0, 1.0/16.0,
        1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0,
    1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0,
    1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0,
    1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0

};
*/
/*
float kernel[49] = {
0.000002,	0.000052,	0.000348,	0.000653,	0.000348,	0.000052,	0.000002,
0.000052,	0.001278,	0.008539,	0.016014,	0.008539,	0.001278,	0.000052,
0.000348,	0.008539,	0.057042,	0.106976,	0.057042,	0.008539,	0.000348,
0.000653,	0.016014,	0.106976,	0.20062,	0.106976,	0.016014,	0.000653,
0.000348,	0.008539,	0.057042,	0.106976,	0.057042,	0.008539,	0.000348,
0.000052,	0.001278,	0.008539,	0.016014,	0.008539,	0.001278,	0.000052,
0.000002,	0.000052,	0.000348,	0.000653,	0.000348,	0.000052,	0.000002 };
*/
glm::ivec2 offset[25] = { {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},
                         {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
                         {-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0},
                        {-2, 1}, {-1, 1}, {0, 1}, {1, 1}, {2, 1},
                        {-2, 2}, {-1, 2}, {0, 2}, {1, 2}, {2, 2} };



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
static GBufferPixel* dev_gBuffer = NULL;
static float* dev_kernel = NULL;
static glm::ivec2* dev_offset = NULL;
static glm::vec3* dev_out = NULL;
static glm::vec3* dev_in = NULL;


//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
    int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO2(uchar4* pbo, glm::ivec2 resolution,
    int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 color = image[index];

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        float timeToIntersect = gBuffer[index].t * 256.0;
        glm::vec3 pos = gBuffer[index].position;
        glm::vec3 nor = gBuffer[index].normal;
        //glm::vec3 pos_normalized = glm::normalize(pos);
        glm::ivec3 color;
        // show normals

        color.x = glm::clamp((int)(glm::abs(nor.x) * 255.0), 0, 255);
        color.y = glm::clamp((int)(glm::abs(nor.y) * 255.0), 0, 255);
        color.z = glm::clamp((int)(glm::abs(nor.z) * 255.0), 0, 255);

        // show positions
        /*
        float scalar = 30;
        color.x = glm::clamp((int)(glm::abs(scalar * pos.x)), 0, 255);
        color.y = glm::clamp((int)(glm::abs(scalar * pos.y)), 0, 255);
        color.z = glm::clamp((int)(glm::abs(scalar * pos.z)), 0, 255);
        */

        pbo[index].w = 0;
        pbo[index].x = color.x;//timeToIntersect;
        pbo[index].y = color.y;//timeToIntersect;
        pbo[index].z = color.z;//timeToIntersect;
    }
}

__global__ void colorPerIter(glm::ivec2 resolution, glm::vec3* image, glm::vec3* colorPerIter, int iter) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        colorPerIter[index].x = color.x;
        colorPerIter[index].y = color.y;
        colorPerIter[index].z = color.z;
    }
}


__global__ void denoise(glm::ivec2 resolution, int iter, glm::vec3* image, glm::ivec2* offset, float* kernel, GBufferPixel* gBuffer, int stepwidth, glm::vec3* out, float c_phi, float n_phi, float p_phi) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        //uchar4 color = pbo[index];
        glm::vec3 cval = image[index] / 255.0f;

        glm::vec3 sum(0.0f);
        //glm::ivec2 step((int) (1.0f / (float)resolution.x), (int) (1.0f / (float)resolution.y));

        glm::vec3 pval = gBuffer[index].position;
        glm::vec3 nval = gBuffer[index].normal;
        float sampleFrame = stepwidth;
        float sf2 = sampleFrame * sampleFrame;

        float cum_w = 0.0f;
        //c_phi = 0.572f;
        //n_phi = 0.021f;
        //p_phi = 0.789f;


            for (int i = 0; i < 25; i++) {

                glm::ivec2 uv;

                uv.x = glm::clamp(x + offset[i].x * stepwidth, 0, resolution.x - 1);
                uv.y = glm::clamp(y + offset[i].y * stepwidth, 0, resolution.y - 1);


                int point_index = uv.x + (uv.y * resolution.x);
                
                glm::vec3 ctmp = image[point_index] / 255.0f;

                glm::vec3 t = cval - ctmp;
                float dist2 = glm::dot(t, t);
                float c_w = glm::min(glm::exp(-(dist2) / c_phi), 1.0f);

                glm::vec3 ntmp = gBuffer[point_index].normal;
                t = nval - ntmp;
                dist2 = glm::max(glm::dot(t, t) / (stepwidth * stepwidth), 0.0f);
                float n_w = glm::min(glm::exp(-(dist2) / n_phi), 1.0f);

                glm::vec3 ptmp = gBuffer[point_index].position;
                t = pval - ptmp;
                dist2 = glm::dot(t, t);
                float p_w = glm::min(glm::exp(-(dist2) / p_phi), 1.0f);
#if GAUSSIAN
                float weight = kernel[i];
#else
                float weight = c_w * n_w * p_w * kernel[i];

#endif // GAUSSIAN


                sum += ctmp * weight;
                cum_w += weight;

            }
            sum *= 255.0f;
            out[index] = sum / cum_w;
    }
}
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

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    cudaMalloc(&dev_kernel, 25 * sizeof(float));
    cudaMemcpy(dev_kernel, kernel, 25 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_offset, 25 * sizeof(glm::ivec2));
    cudaMemcpy(dev_offset, offset, 25 * sizeof(glm::ivec2), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_out, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_out, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_in, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_in, 0, pixelcount * sizeof(glm::vec3));


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
    cudaFree(dev_gBuffer);
    cudaFree(dev_kernel);
    cudaFree(dev_offset);
    cudaFree(dev_out);
    cudaFree(dev_in);
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
__global__ void shadeMaterial (
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


__global__ void generateGBuffer(
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    GBufferPixel* gBuffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        gBuffer[idx].t = shadeableIntersections[idx].t;
        gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
        gBuffer[idx].position = getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t);
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
void pathtrace(int frame, int iter) {
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
    if (depth == 1) {
        generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
    }
    shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
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
    //sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(hst_scene->state.image_denoise.data(), dev_out, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}


// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
}

void showImage(uchar4* pbo, int iter, bool isdenoise, float c_phi, float n_phi, float p_phi, int filter_size) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    int pixelcount = cam.resolution.x * cam.resolution.y;

    int stepwidth = 1;
    double time = 0.0f;
    if (isdenoise) {
        colorPerIter << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, dev_image, dev_in, iter);

        for (stepwidth = 1; stepwidth <= filter_size / 2; stepwidth *= 2) {
            //printf("Denoise step: %d\n", stepwidth);
            auto start = std::chrono::steady_clock::now();

            denoise << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, iter, dev_in, dev_offset, dev_kernel, dev_gBuffer, stepwidth, dev_out, c_phi, n_phi, p_phi);

            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            time += elapsed_seconds.count();
            cudaMemcpy(dev_in, dev_out, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
            //dev_in = dev_out;
        }
        std::cout << "filter_size:" << filter_size << ",   elapsed time: " <<  time << "s\n";

        sendImageToPBO2 << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_out);


    }
    else {
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
    }
}