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
void print_matrix(double* matrix, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf("%lf ", matrix[i * cols + j]);
		}
		printf("\n");
	}
	printf("\n");
}
void copy_matrix(double* original, double*copy, int rows, int cols){
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			copy[i*cols + j] = original[i*cols + j];
		}
	}
}
bool hasConverged(double* newCentroids, double*oldCentroids, int rows, int cols){
	float tolerance = 0.001;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (abs(oldCentroids[i*cols + j] - newCentroids[i*cols + j]) > tolerance){
				return false;
			}
		}
	}
	return true;
}

__global__ void registerPointsToCentroid(double* points, double* centroids, int* linkMatrix, int numPoints, int numCentroids, int numAttributes){
	int numThread = blockIdx.x * blockDim.x + threadIdx.x;

	if (numThread < numPoints){//allora vuol dire che il thread corrisponde ad un Point in memoria
		//assignPointToCloserCentroid(points, sharedCentroids, linkMatrix, numPoints, numCentroids, numAttributes, numThread);

		double minDist = DBL_MAX;
		int closerCentroidID;
		double currentDist;

		for (int i = 0; i < numCentroids; i++){ //itero per ogni centroide
			currentDist = 0.0;
			for (int j = 0; j < numAttributes; j++){//riempio il vettore con la posizione del centroide
				currentDist = currentDist + pow((points[numThread*numAttributes + j] - centroids[i*numAttributes + j]), 2.0);
			}
			//currentDist = sqrt(currentDist); potrebbe non servire ?
			if (currentDist < minDist){
				minDist = currentDist;
				closerCentroidID = i;
			}
		} //alla fine di questo for ho l'ID del centroide più vicino - questo va registrato sulla matrice dei link
		//printf("\ncloserCentroi ID : %d", closerCentroidID);
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

	double *ptsMatrix; //matrice dei punti (numPoints x numAttributes)
	double *centersMatrix; //matrice dei centroidi (numCentroids x numAttributes)
	int *linkMatrix; //matrice che descrive l'appartenenza di punti a centroidi (numCentroids x numPoints)
	double *oldCentersMatrix; //serve per capire quando si ha convergenza
	int *vectorIndex; //serve per il ricalcolo dei centroidi

	double *devicePtsMatrix; // versione su device
	double *deviceCentersMatrix; //versione su device
	int *deviceLinkMatrix; //versione su device

	int numAttributes;
	int numPoints;
	int numCentroids;


	FileReader* reader = new FileReader();
	(*reader).readFile(pointsPath, &points);
	(*reader).readFile(centersPath, &centroids);

	numAttributes = points.at(0).numAttributes;
	numPoints = points.size();
	numCentroids = centroids.size();

	ptsMatrix = (double*)malloc(numPoints*numAttributes*sizeof(double));
	centersMatrix = (double*)malloc(numCentroids*numAttributes*sizeof(double));
	linkMatrix = (int*)malloc(numCentroids*numPoints*sizeof(int));

	oldCentersMatrix = (double*)malloc(numCentroids*numAttributes*sizeof(double));

	//printf("copio i punti su matrici");
	int start_s = clock();
	//inizializzo la matrice di punti
	for (int i = 0; i < numPoints; i++){
		for (int j = 0; j < numAttributes; j++){
			ptsMatrix[i*numAttributes + j] = points.at(i).attributes[j];
			//printf("  %lf", points.at(i).attributes[j]);
		}
		//printf("\n");
	}
	//inizializzo la matrice di centroidi 
	for (int i = 0; i < numCentroids; i++){
		for (int j = 0; j < numAttributes; j++){
			centersMatrix[i*numAttributes + j] = centroids.at(i).attributes[j];
		}
	}
	// alloco la memoria sul device
	CUDA_CHECK_RETURN(
		cudaMalloc((void **)&devicePtsMatrix, sizeof(double)* numPoints * numAttributes)
		);
	//copio i dati dall'host alla memoria del device
	CUDA_CHECK_RETURN(//poi devo provare a metterli nella constant memory
		cudaMemcpy(devicePtsMatrix, ptsMatrix, sizeof(double)* numPoints * numAttributes, cudaMemcpyHostToDevice)
		);



	do{
		printf("\n iterazione numero: %d \n", iterationNum);
		//inizializzo la matrice di link
		for (int i = 0; i < numPoints; i++){
			for (int j = 0; j < numCentroids; j++){
				linkMatrix[i*numCentroids + j] = 0;
			}
		}
		copy_matrix(centersMatrix, oldCentersMatrix, numCentroids, numAttributes);

		// alloco la memoria sul device
		CUDA_CHECK_RETURN( 
			cudaMalloc((void **)&deviceCentersMatrix, sizeof(double)* numCentroids * numAttributes)
			);
		CUDA_CHECK_RETURN( 
			cudaMalloc((void **)&deviceLinkMatrix, sizeof(int)* numCentroids * numPoints)
			);

		//copio i dati dall'host alla memoria del device
		CUDA_CHECK_RETURN(
			cudaMemcpy(deviceCentersMatrix, centersMatrix, sizeof(double)* numCentroids * numAttributes, cudaMemcpyHostToDevice)
			);
		CUDA_CHECK_RETURN(//poi devo provare a metterli nella constant memory
			cudaMemcpy(deviceLinkMatrix, linkMatrix, sizeof(int)* numCentroids * numPoints, cudaMemcpyHostToDevice)
			);

		dim3 blockDim(1024, 1);
		dim3 gridDim(ceil(numPoints / 1024.0), 1);

		//esegue kernel
		registerPointsToCentroid << <gridDim, blockDim >> >(devicePtsMatrix, deviceCentersMatrix, deviceLinkMatrix, numPoints, numCentroids, numAttributes);

		cudaDeviceSynchronize();

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
		//azzero l'indice del numero di punti per centroide
		vectorIndex = (int*)malloc(numCentroids*sizeof(int));
		for (int i = 0; i < numCentroids; i++){
			vectorIndex[i] = 0;
		}

		//sommo i valori delle posizioni di ogni punto sul corrispondente centroide
		for (int i = 0; i < numCentroids; i++){
			for (int j = 0; j < numPoints; j++){
				if (linkMatrix[i + numCentroids*j] == 1){//il punto j-esimo appartiene al centroide i-esimo
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
		print_matrix(centersMatrix, numCentroids, numAttributes);

	} while (iterationNum<100 && !hasConverged(centersMatrix, oldCentersMatrix, numCentroids, numAttributes));
	int stop_s = clock();
	std::cout << "time__: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) << std::endl;
	// prova a vedere cosa succede se non svuoti le memorie del device di quei campi che restano uguali (tipo i points) e se non glieli ripassi, se va uguale risparmi tempo


	cudaFree(devicePtsMatrix);
	free(ptsMatrix);
	free(centersMatrix);
	free(linkMatrix);


	return 0;
}

