#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__global__ void dkernel(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k)
{
    //used dynamic shared memory to bring filter into shared memory
    extern __shared__ long int s_filter[];
    
    //loading shared memory through copying from global filter
    unsigned tid=threadIdx.x;
    
    //here used coalescing like every thread call consecutive location at a time and second instruction for same thread call location+1024th place
    for(int p=0;p<ceil((r*s*c*k*1.0)/1024.0);p++){
        if(1024*p+tid<r*s*c*k){
            s_filter[p*1024+tid]=filter[p*1024+tid];
        }
    }
    
    // barrier for complete filter load in shared memory
    __syncthreads();
    
    //find unique id
	unsigned id= blockIdx.x * blockDim.x + threadIdx.x;
	
	//check for last block
	if (id< h*w*k){
	    
	    //for computation
 		ll ans=0;
 		
 		//for  finding matrix id
		int m_id=id%(h*w);
		
		// find filter number like total h*w*k thread so find k
		int f_num = id/(h*w);
		
		//find row number
		int x=m_id/w;
		
		//find column number
		int y= m_id%w;
		
		//for all channels
		for(int i=0;i<c;i++){
		    
		    //for filter's all row and its always odd so direcct devide by 2 and -r/2 to r/2 apply filter to center in row
			for(int j=-r/2;j<=r/2;j++){
			    
			    //for filter's all column and its always odd so direcct devide by 2 and -s/2 to s/2 apply filter to center in column
				for(int l= -s/2 ;l<=s/2 ;l++){
				
				    //for matrix index check for outofbound or say padding x+j shoud be less than  total row and y+l not exceed total column
					if(x+j>=0 && x+j<h && y+l>=0 && y+l<w){
					
					    //matrix index using channel ,according to filter row and column
					    // filter used shared memory and index using filter number,channel ,row, column add in ans
						ans+= matrix[(h*w*i)+((x+j)*w+(y+l))] * s_filter[(f_num*r*s*c)+(i*r*s)+(j+r/2)*s+(l+s/2)];
					}
				}
			}		
		}
		
		//finally after computing update final ans in result		
		result[id]=ans;
		
	}

	

	
}

int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++)
    {
        cin >> h_mat[i];
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++)
    {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     **/

    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch

    /****************************************************Start Here***********************************************************/
	
    
    // memory allocation for kernel
	long int *p_ans;
	long int *p_mat;
	long int *p_filter;
	cudaMalloc(&p_ans, h*w*k*sizeof(long int));
	cudaMalloc(&p_mat, h*w*c*sizeof(long int));
	cudaMalloc(&p_filter, r*s*c*k*sizeof(long int));
	
	    //memory copy to bring it to gpu kernel
        cudaMemcpy(p_mat, h_mat, h * w * c *sizeof(long int), cudaMemcpyHostToDevice);
        cudaMemcpy(p_filter, h_filter, r * s * c * k * sizeof(long int), cudaMemcpyHostToDevice);

    //total number of block 
    //here i use total thread h*w*k because answer matrix size is h*w*k if i use more than have to use syncronization
	unsigned block = ceil(h*w*k*1.0/1024.0);
      
      //here i used preferEqual means 32KB for shared memory like maximum constraint for filter size is 4096 so
      // total maximum size of filter is 4k * 8 Byte(sizeof(long int) so 32KB require 
      //so other 32kb can be utilize by L1 cache
      cudaFuncSetCacheConfig(dkernel, cudaFuncCachePreferEqual); 
      
      
    // kernel launch with block * 1024 thread, and  dynamic shared memory for storing filter into it  
	dkernel<<< block,1024, (r*s*c*k* sizeof(long int))>>> (p_mat,p_filter,p_ans,h,w,c,r,s,k);

    // answer copying from device to host
	cudaMemcpy(h_ans,p_ans, h*w*k * sizeof(long int),cudaMemcpyDeviceToHost);

	cudaFree(p_mat);
	cudaFree(p_filter);
	cudaFree(p_ans);

    //done

    /**
        Do device allocations, kernel launches and copying everything here
        and the final answer should be stored back in h_ans, use cudaFree to free up the allocated memory on GPU
    */

    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */

    cudaDeviceSynchronize();
    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
               

            }
            file << "\n";
            
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}

