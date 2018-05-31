/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance- or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.Luminance
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include<stdio.h>

__global__
void sumlum(const float* d_in,float* d_intermediate){
	//shared memory
	extern __shared__ float sdata[];
	unsigned int idx=threadIdx.x+blockDim.x*blockIdx.x;
	unsigned int tid=threadIdx.x;
	
	sdata[tid]=d_in[idx];
	__syncthreads();
	for(unsigned int s=blockDim.x/2;s>0;s>>=1){
		if(tid<s){
			sdata[tid]+=sdata[tid+s];}
		__syncthreads();}
	if(tid==0){
		d_intermediate[blockIdx.x]=sdata[0];}
		}

__global__
void minlum(const float* const d_in,float* d_intermediate){
	//shared memory
	extern __shared__ float sdata[];
	//indices
	unsigned int idx=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int tid=threadIdx.x;
	
	sdata[tid]=d_in[idx];
	__syncthreads();
	for(unsigned int s=blockDim.x/2;s>0;s>>=1){
		if(tid<s){
			sdata[tid]=min(sdata[tid+s],sdata[tid]);
			}
		__syncthreads();	
		}
	if(tid==0){
	d_intermediate[blockIdx.x]=sdata[0];
		}
	}

__global__	
void maxlum(const float* const d_in,float* d_intermediate,size_t numRows,size_t numCols){
	//shared memory
	extern __shared__ float sdata[];
	//indices
	
	unsigned int idx=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int tid=threadIdx.x;

	unsigned index=numRows*numCols;

	if(idx>=index){
		return;}
	
	sdata[tid]=d_in[idx];
	__syncthreads();
	
	for(unsigned int s=blockDim.x/2;s>0;s>>=1){
		if(tid<s && tid+s< numCols){
			sdata[tid]=max(sdata[tid+s],sdata[tid]);
			}
		__syncthreads();	
		}
	if(tid==0){
	d_intermediate[blockIdx.x]=sdata[0];
		}
	}
	
__global__
void histo(unsigned int* const d_bins,const float* const d_in,const size_t NUM_BINS,int lumMax,int lumMin ){
	int idx=threadIdx.x+blockDim.x*blockIdx.x;

	int myBin=((d_in[idx]-lumMin)/(lumMax-lumMin))*NUM_BINS;
	atomicAdd(&(d_bins[myBin]),1);
}


__global__
void steeleScan(unsigned int* const d_bins,const size_t BIN_COUNT){
	//Hillis Steele Inclusive scan
	unsigned int idx=threadIdx.x+blockDim.x*blockIdx.x;
	extern __shared__ float sdata[];
	
	sdata[idx]=d_bins[idx];
	__syncthreads();
	for(unsigned int s=1;s<BIN_COUNT;s<<=1){
		if(idx>=s){
			sdata[idx]+=sdata[idx-s];
		}
		__syncthreads();}
	d_bins[idx]=sdata[idx]-d_bins[idx];}
		
float *dmin_inter,*dmax_inter,*dmin_final,*dmax_final;

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{

    dim3 blockSize(numCols);
    dim3 gridSize(numRows);
    //allocate memory for intermediate and final
    checkCudaErrors(cudaMalloc(&dmax_inter,sizeof(float)*numRows));
    checkCudaErrors(cudaMalloc(&dmax_final,sizeof(float)));
    //begin two stage reduction
    maxlum<<<gridSize,blockSize,blockSize.x*sizeof(float)>>>(d_logLuminance,dmax_inter,numRows,numCols);
    cudaDeviceSynchronize();checkCudaErrors(cudaGetLastError());
    maxlum<<<1,gridSize,gridSize.x*sizeof(float)>>>(dmax_inter,dmax_final,numRows,numCols);
    cudaDeviceSynchronize();checkCudaErrors(cudaGetLastError());
    //copy to host
    cudaMemcpy(&max_logLum,dmax_final,sizeof(float),cudaMemcpyDeviceToHost);
    
    //repeat max procedure for min
    checkCudaErrors(cudaMalloc(&dmin_inter,sizeof(float)*numRows));
    checkCudaErrors(cudaMalloc(&dmin_final,sizeof(float)));
    minlum<<<gridSize,blockSize,blockSize.x*sizeof(float)>>>(d_logLuminance,dmin_inter);
    cudaDeviceSynchronize();
    minlum<<<1,gridSize,gridSize.x*sizeof(float)>>>(dmin_inter,dmin_final);
    cudaDeviceSynchronize();
    cudaMemcpy(&min_logLum,dmin_final,sizeof(float),cudaMemcpyDeviceToHost);
    
    histo<<<gridSize,blockSize>>>(d_cdf,d_logLuminance,numBins,max_logLum,min_logLum);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    steeleScan<<<1,numBins,numBins*sizeof(float)>>>(d_cdf,numBins);
    cudaDeviceSynchronize();checkCudaErrors(cudaGetLastError());
    cudaFree(dmin_inter);
    cudaFree(dmax_inter);
    cudaFree(dmin_final);
    cudaFree(dmax_final);
    
}
