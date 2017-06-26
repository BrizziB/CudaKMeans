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

__global__ void registerPointsToCentroid(float* points, float* centroids, int* linkMatrix, int numPoints, int numCentroids, int numAttributes, int centroidTilingSize){
	int numThread = blockIdx.x * blockDim.x + threadIdx.x;
	
	extern __shared__ float sharedCentroids[];

	if (numThread < numPoints){//allora vuol dire che il thread corrisponde ad un Point in memoria

		float minDist = FLT_MAX;
		int closerCentroidID;
		float currentDist = 0.0;

		for (int i = 0; i < floor(( (double)numCentroids / (double)centroidTilingSize )) ; i++){ //itero per ogni centroide
			//tiling dei centroidi----------------------------------------------------------------------------------------------
			if (threadIdx.x < centroidTilingSize*numAttributes){
				sharedCentroids[threadIdx.x] = centroids[i*centroidTilingSize*numAttributes + threadIdx.x];
			}
			__syncthreads();
			//fine tiling------------------------------------------------------------------------------------------------------
			for (int k = 0; k < centroidTilingSize; k++){
				currentDist = 0.0;
				for (int j = 0; j < numAttributes; j++){//riempio il vettore con la posizione del centroide
					currentDist = currentDist + pow(((double)points[numThread*numAttributes + j] - (double)sharedCentroids[k*numAttributes + j]), 2.0);
				}
				if (currentDist < minDist){
					minDist = currentDist;
					closerCentroidID = k + i*centroidTilingSize;
				}
			}
		}
		int index = floor(((double)numCentroids / (double)centroidTilingSize));
		//tiling dei centroidi rimanenti - se presenti --------------------------------------------------------------------
		if (threadIdx.x < (numCentroids - index*centroidTilingSize)*numAttributes){
			sharedCentroids[threadIdx.x] = centroids[index*centroidTilingSize*numAttributes + threadIdx.x];
		}
		__syncthreads();
		//fine tiling------------------------------------------------------------------------------------------------------
		for (int k = 0; k < numCentroids - centroidTilingSize*index; k++){
			currentDist = 0.0;
			for (int j = 0; j < numAttributes; j++){//riempio il vettore con la posizione del centroide
				currentDist = currentDist + pow(((double)points[numThread*numAttributes + j] - (double)sharedCentroids[k*numAttributes + j]), 2.0);
			}
			if (currentDist < minDist){
				minDist = currentDist;
				closerCentroidID = centroidTilingSize*index + k;
			}
		}
		//alla fine di questo for ho l'ID del centroide più vicino - questo va registrato sulla matrice dei link
		linkMatrix[numThread*numCentroids + closerCentroidID] = 1;
	}
	//l'aggiornamento dei centroidi si fa su host, non riesco a vedere come si potrebbe risparmiare facendolo qui
}

int main(){
	std::string pointsPath = "dataset.txt";
	std::string centersPath = "centers.txt";
	std::vector<Point> points{};
	std::vector<Centroid> centroids{};

	int iterationNum = 0;
	int toleratedFraction = 1000; //più è alto più è piccolo il numero di punti che può cambiare per dichiarare la convergenza
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

	int numAttributes;
	int numPoints;
	int numCentroids;
	int centroidTilingSize = 10; //QUESTO DEVE SEMPRE ESSERE MINORE AL NUMERO DEI CENTROIDI!

	FileReader* reader = new FileReader();
	(*reader).readFile(pointsPath, &points); //leggo i punti da file
	(*reader).readFile(centersPath, &centroids);

	numAttributes = points.at(0).numAttributes;
	numPoints = points.size();
	numCentroids = centroids.size();

	ptsMatrix = (float*)malloc(numPoints*numAttributes*sizeof(float));
	centersMatrix = (float*)malloc(numCentroids*numAttributes*sizeof(float));
	linkMatrix = (int*)malloc(numCentroids*numPoints*sizeof(int));
	oldLinkMatrix = (int*)malloc(numCentroids*numPoints*sizeof(int));;

	int start_s = clock(); //registro tempi
	//inizializzo la matrice di punti
	for (int i = 0; i < numPoints; i++){
		for (int j = 0; j < numAttributes; j++){
			ptsMatrix[i*numAttributes + j] = points.at(i).attributes[j];
		}
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
		//inizializzo la nuova matrice di link e la copio sulla vecchia
		for (int i = 0; i < numPoints; i++){
			for (int j = 0; j < numCentroids; j++){

				if (iterationNum == 0){
					oldLinkMatrix[i*numCentroids + j] = 0;
				}
				else{
					oldLinkMatrix[i*numCentroids + j] = linkMatrix[i*numCentroids + j];
				}
				linkMatrix[i*numCentroids + j] = 0;
			}
		}
		for (int i = 0; i < numCentroids; i++){
			vectorIndex[i] = 0;
		}
		// alloco la memoria sul device
		CUDA_CHECK_RETURN(
			cudaMalloc((void **)&deviceCentersMatrix, sizeof(float)* numCentroids * numAttributes)
			);
		CUDA_CHECK_RETURN(
			cudaMalloc((void **)&deviceLinkMatrix, sizeof(int)* numCentroids * numPoints)
			);
		//copio i dati dall'host alla memoria del device
		CUDA_CHECK_RETURN(
			cudaMemcpy(deviceCentersMatrix, centersMatrix, sizeof(float)* numCentroids * numAttributes, cudaMemcpyHostToDevice)
			);
		CUDA_CHECK_RETURN(//poi devo provare a metterli nella constant memory
			cudaMemcpy(deviceLinkMatrix, linkMatrix, sizeof(int)* numCentroids * numPoints, cudaMemcpyHostToDevice)
			);

		dim3 blockDim(1024, 1);
		dim3 gridDim(ceil(numPoints / 1024.0), 1);


		//print_matrix(centersMatrix, numCentroids, numAttributes);
		//-------------------------------esegue kernel----------------------------------------------------------
		registerPointsToCentroid << <gridDim, blockDim, centroidTilingSize*numAttributes*sizeof(float) >> >
			(devicePtsMatrix, deviceCentersMatrix, deviceLinkMatrix, numPoints, numCentroids, numAttributes, centroidTilingSize);
		cudaDeviceSynchronize();
		//-------------------------------esegue host------------------------------------------------------------
		
		//copio dati da device a host
		CUDA_CHECK_RETURN(
			cudaMemcpy(linkMatrix, deviceLinkMatrix, numCentroids*numPoints*sizeof(int), cudaMemcpyDeviceToHost)
			);

		//svuoto le memorie del device
		cudaFree(deviceCentersMatrix);
		cudaFree(deviceLinkMatrix);

		//ricalcolo i centroidi
		//azzero la matrice dei centroidi
		for (int i = 0; i < numCentroids; i++){
			for (int j = 0; j < numAttributes; j++){ 
				centersMatrix[i*numAttributes + j] = 0;
			}
		}
		numPtsChanged = 0;
		//sommo i valori delle posizioni di ogni punto sul corrispondente centroide
		for (int i = 0; i < numCentroids; i++){
			for (int j = 0; j < numPoints; j++){
				if (linkMatrix[i + numCentroids*j] == 1){//il punto j-esimo appartiene al centroide i-esimo
					if (oldLinkMatrix[i + numCentroids*j] == 0){//per controllo della convergenza
						numPtsChanged++;
					}
					for (int k = 0; k < numAttributes; k++){
						centersMatrix[i*numAttributes + k] += ptsMatrix[j*numAttributes + k];
					}
					vectorIndex[i]++;//registro il numero di punti per centroide
				}
			}
		}

		//faccio la media usando vectorIndex
		for (int i = 0; i < numCentroids; i++){
			for (int j = 0; j < numAttributes; j++){
				centersMatrix[i*numAttributes + j] = centersMatrix[i*numAttributes + j] / vectorIndex[i];
			}
		}
		iterationNum++;
		printf("numero punti cambiati: %d", numPtsChanged);
		convergence = hasConverged(numPtsChanged, numPoints, toleratedFraction);
	} while (iterationNum<100 && !convergence);


	int stop_s = clock();
	std::cout << "\ntime: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) << std::endl;
	// prova a vedere cosa succede se non svuoti le memorie del device di quei campi che restano uguali (tipo i points) e se non glieli ripassi, se va uguale risparmi tempo

	print_matrix(centersMatrix, numCentroids, numAttributes);
	cudaFree(devicePtsMatrix);
	free(ptsMatrix);
	free(centersMatrix);
	free(linkMatrix);
	free(vectorIndex);
	free(oldLinkMatrix);
	
	return 0;
}

