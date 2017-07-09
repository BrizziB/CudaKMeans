#include <iostream>
#include <numeric>
#include <string>
#include <fstream>
#include <regex>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include "Point.h"
#include "Centroid.h"
#include "FileReader.h"

static void CheckCudaErrorAux(const char *, unsigned, const char *,
	cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
* Check the return value of the CUDA runtime API call and exit
* the application if the call has failed.
*/
static void CheckCudaErrorAux(const char *file, unsigned line,
	const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
		<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}
void print_matrix(int* matrix, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf("%d ", matrix[i * cols + j]);
		}
		printf("\n");
	}
	printf("\n");
}
void print_matrix(float* matrix, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf("%lf ", matrix[i * cols + j]);
		}
		printf("\n");
	}
	printf("\n");
}
void copy_matrix(float* original, float*copy, int rows, int cols){
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			copy[i*cols + j] = original[i*cols + j];
		}
	}
}
bool hasConverged(int numPtsChanged, int numPts, int toleratedFraction){
	printf("\nsoglia: %d", numPts / toleratedFraction);
	if (numPtsChanged > numPts / toleratedFraction)
		return false;
	return true;

}


__global__ void registerPointsToCentroid_tiling(float* points, float* centroids, int* linkMatrix, int numPoints, int numCentroids, int numAttributes, int centroidTilingSize){
	int numThread = blockIdx.x * blockDim.x + threadIdx.x;

	// ---------------------------------------------------- VERSIONE CON TILING -----------------------------------------------------------------------
	extern __shared__ float sharedCentroids[];
	if (numThread < numPoints){//allora vuol dire che il thread corrisponde ad un Point in memoria
		float minDist = FLT_MAX;
		int closerCentroidID;
		float currentDist;
		for (int i = 0; i < floor(((double)numCentroids / (double)centroidTilingSize)); i++){ //itero per ogni centroide
			//tiling of centroids----------------------------------------------------------------------------------------------
			if (threadIdx.x < centroidTilingSize*numAttributes){
				sharedCentroids[threadIdx.x] = centroids[i*centroidTilingSize*numAttributes + threadIdx.x];
			}
			//end of tiling------------------------------------------------------------------------------------------------------
			for (int k = 0; k < centroidTilingSize; k++){
				__syncthreads();
				currentDist = 0.0;
				for (int j = 0; j < numAttributes; j++){
					currentDist = currentDist + pow(((double)points[numThread*numAttributes + j] - (double)sharedCentroids[k*numAttributes + j]), 2.0);
				}
				if (currentDist < minDist){
					minDist = currentDist;
					closerCentroidID = k + i*centroidTilingSize;
				}
			}


		}
		int index = floor(((double)numCentroids / (double)centroidTilingSize));
		//tiling of remaining centroids - if presents
		if (threadIdx.x < (numCentroids - index*centroidTilingSize)*numAttributes){
			sharedCentroids[threadIdx.x] = centroids[index*centroidTilingSize*numAttributes + threadIdx.x];
		}
		//end of tiling--------------------------
		for (int k = 0; k < numCentroids - centroidTilingSize*index; k++){
			__syncthreads();
			currentDist = 0.0;
			for (int j = 0; j < numAttributes; j++){
				currentDist = currentDist + pow(((double)points[numThread*numAttributes + j] - (double)sharedCentroids[k*numAttributes + j]), 2.0);
			}
			if (currentDist < minDist){
				minDist = currentDist;
				closerCentroidID = centroidTilingSize*index + k;
			}
		}
		linkMatrix[numThread] = closerCentroidID;
	}
}

__global__ void registerPointsToCentroid(float* points, float* centroids, int* linkMatrix, int numPoints, int numCentroids, int numAttributes){
	int numThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (numThread < numPoints){//then the thread is associated to a point in the global memory
		float minDist = FLT_MAX;
		int closerCentroidID;
		float currentDist;
		for (int i = 0; i < numCentroids; i++){ //iterating for all the centroids
			currentDist = 0.0;
			for (int j = 0; j < numAttributes; j++){
				currentDist = currentDist + pow(((double)points[numThread*numAttributes + j] - (double)centroids[i*numAttributes + j]), 2.0);
			}
			if (currentDist < minDist){
				minDist = currentDist;
				closerCentroidID = i;
			}
		} 
		linkMatrix[numThread] = closerCentroidID;
	}
}

int main(){
	std::string pointsPath = "dataset.txt";
	std::string centersPath = "centers.txt";
	std::vector<Point> points{};
	std::vector<Centroid> centroids{};

	int iterationNum = 0;
	int toleratedFraction = 500;
	int numPtsChanged;
	bool convergence = false;

	float *ptsMatrix; //matrice dei punti (numPoints x numAttributes)
	float *centersMatrix; //matrice dei centroidi (numCentroids x numAttributes)
	int *linkMatrix; //matrice che descrive l'appartenenza di punti a centroidi (numCentroids x numPoints)
	int *vectorIndex; //serve per il ricalcolo dei centroidi
	int *oldLinkMatrix; //serve per il calcolo della convergenza

	float *devicePtsMatrix; // versione su device
	float *deviceCentersMatrix; //versione su device
	int *deviceLinkMatrix; //versione su device

	bool tilingEnabled = false;


	int numAttributes;
	int numPoints;
	int numCentroids;
	int centroidTilingSize;

	FileReader* reader = new FileReader();
	(*reader).readFile(pointsPath, &points); //leggo i punti da file
	(*reader).readFile(centersPath, &centroids);

	numAttributes = points.at(0).numAttributes;
	numPoints = points.size();
	numCentroids = centroids.size();

	ptsMatrix = (float*)malloc(numPoints*numAttributes*sizeof(float));
	centersMatrix = (float*)malloc(numCentroids*numAttributes*sizeof(float));
	linkMatrix = (int*)malloc(numPoints*sizeof(int));
	oldLinkMatrix = (int*)malloc(numPoints*sizeof(int));;
 
	if (numCentroids*numAttributes <= 500 ){
		centroidTilingSize = numCentroids;
	}
	else{
		centroidTilingSize = 500;
	}
	

	int start_s = clock(); //registro tempi
	//inizializzo la matrice di punti e il vettore di link
	for (int i = 0; i < numPoints; i++){
		for (int j = 0; j < numAttributes; j++){
			ptsMatrix[i*numAttributes + j] = points.at(i).attributes[j];
		}
		linkMatrix[i] = -1;
	}
	//inizializzo la matrice di centroidi 
	for (int i = 0; i < numCentroids; i++){
		for (int j = 0; j < numAttributes; j++){
			centersMatrix[i*numAttributes + j] = centroids.at(i).attributes[j];
		}
	}
	// alloco la memoria sul device
	CUDA_CHECK_RETURN(
		cudaMalloc((void **)&devicePtsMatrix, sizeof(float)* numPoints * numAttributes)
		);
	//copio i Punti  dall'host alla memoria del device
	CUDA_CHECK_RETURN(//poi devo provare a metterli nella constant memory
		cudaMemcpy(devicePtsMatrix, ptsMatrix, sizeof(float)* numPoints * numAttributes, cudaMemcpyHostToDevice)
		);

	//alloco la memoria necessaria e copio il vettore di link
	CUDA_CHECK_RETURN(
		cudaMalloc((void **)&deviceLinkMatrix, sizeof(int)* numPoints)
		);
	CUDA_CHECK_RETURN(
		cudaMemcpy(deviceLinkMatrix, linkMatrix, sizeof(int)* numPoints, cudaMemcpyHostToDevice)
		);


	//azzero l'indice del numero di punti per centroide
	vectorIndex = (int*)malloc(numCentroids*sizeof(int));
	for (int i = 0; i < numCentroids; i++){
		vectorIndex[i] = 0;
	}
	// ---------------------------------------------------------------------------------------------------------- 
	// ----------------------------------------- inizia ciclo principale ----------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	do{
		printf("\n iterazione esterna numero: %d \n", iterationNum);
		//salve il vettore di link della iterazione precedente
		for (int i = 0; i < numPoints; i++){
			oldLinkMatrix[i] = linkMatrix[i];
		}
		for (int i = 0; i < numCentroids; i++){
			vectorIndex[i] = 0;
		}

		// alloco la memoria sul device
		CUDA_CHECK_RETURN(
			cudaMalloc((void **)&deviceCentersMatrix, sizeof(float)* numCentroids * numAttributes)
			);
		//copio i centroidi dall'host alla memoria del device
		CUDA_CHECK_RETURN(
			cudaMemcpy(deviceCentersMatrix, centersMatrix, sizeof(float)* numCentroids * numAttributes, cudaMemcpyHostToDevice)
			);

		dim3 blockDim(1024, 1);
		dim3 gridDim(ceil(numPoints / 1024.0), 1);

		//print_matrix(centersMatrix, numCentroids, numAttributes);

		//-------------------------------esegue kernel----------------------------------------------------------
		if (!tilingEnabled)
			registerPointsToCentroid << <gridDim, blockDim>> >(devicePtsMatrix, deviceCentersMatrix, deviceLinkMatrix, numPoints, numCentroids, numAttributes);
		
		// -- versione con Tiling --
		else{
			registerPointsToCentroid_tiling << <gridDim, blockDim, centroidTilingSize*numAttributes*sizeof(float) >> >(devicePtsMatrix, deviceCentersMatrix, deviceLinkMatrix, numPoints, numCentroids, numAttributes, centroidTilingSize);
		}

		//
		cudaDeviceSynchronize();

		//-------------------------------esegue host------------------------------------------------------------

		//copio dati da device a host
		CUDA_CHECK_RETURN(
			cudaMemcpy(linkMatrix, deviceLinkMatrix, numPoints*sizeof(int), cudaMemcpyDeviceToHost)
			);

		//svuoto le memorie del device - apparte quelle che mantengo per tutte le iterazioni
		cudaFree(deviceCentersMatrix);

		//ricalcolo i centroidi
		//azzero la matrice dei centroidi
		for (int i = 0; i < numCentroids; i++){
			for (int j = 0; j < numAttributes; j++){
				centersMatrix[i*numAttributes + j] = 0;
			}
		}
		numPtsChanged = 0;
		//add position of each point on the right cluster
		int index;
		for (int i = 0; i < numPoints; i++){
			index = linkMatrix[i];
			if (oldLinkMatrix[i] != linkMatrix[i]){//for convergency check
				numPtsChanged++;
			}
			for (int k = 0; k < numAttributes; k++){
				centersMatrix[index*numAttributes + k] += ptsMatrix[i*numAttributes + k];
			}
			vectorIndex[index]++;//tracks the number of points per centroid
		}



		//compute the mean position using vectorIndex
		for (int i = 0; i < numCentroids; i++){
			for (int j = 0; j < numAttributes; j++){
				if (vectorIndex[i]==0)
					centersMatrix[i*numAttributes + j] =-FLT_MAX;
				else
					centersMatrix[i*numAttributes + j] = centersMatrix[i*numAttributes + j] / vectorIndex[i];
			}
		}
		iterationNum++;
		printf("numero punti cambiati: %d", numPtsChanged);
		convergence = hasConverged(numPtsChanged, numPoints, toleratedFraction);
	} while (iterationNum<200 && !convergence);


	int stop_s = clock();
	std::cout << "\ntime: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) << std::endl;

	//print_matrix(centersMatrix, numCentroids, numAttributes);
	cudaFree(deviceLinkMatrix);
	cudaFree(devicePtsMatrix);
	free(ptsMatrix);
	free(centersMatrix);
	free(linkMatrix);
	free(vectorIndex);
	free(oldLinkMatrix);

	return 0;
}

