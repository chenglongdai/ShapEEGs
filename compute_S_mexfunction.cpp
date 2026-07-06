#include "mex.h"
#include <vector>

/* 双精度 (Double) 梯度累加内核 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // 1. 检查
    if (nrhs != 5) mexErrMsgIdAndTxt("MyToolbox:grad:nrhs", "Need 5 inputs.");
    if (!mxIsDouble(prhs[0])) mexErrMsgIdAndTxt("MyToolbox:grad:type", "Inputs must be DOUBLE.");

    // 2. 获取维度
    const mwSize *dim_eeg = mxGetDimensions(prhs[0]);
    int N = (int)dim_eeg[0];
    int K = (int)mxGetM(prhs[1]);
    int L = (int)mxGetN(prhs[1]);
    const mwSize *dim_w = mxGetDimensions(prhs[2]);
    int Q = (int)dim_w[2]; 

    // 3. 获取指针
    double *EEG = mxGetPr(prhs[0]);
    double *Shapelets = mxGetPr(prhs[1]);
    double *W_all = mxGetPr(prhs[2]);
    double *Part1 = mxGetPr(prhs[3]);
    double *Sum_W = mxGetPr(prhs[4]);

    // 4. 输出
    mwSize out_dims[3] = {(mwSize)K, (mwSize)N, (mwSize)L};
    plhs[0] = mxCreateNumericArray(3, out_dims, mxDOUBLE_CLASS, mxREAL);
    double *Grad = mxGetPr(plhs[0]);

    // 5. 计算
    int KN = K * N; 
    for (int k = 0; k < K; ++k) {
        std::vector<double> S_k(L);
        for(int t=0; t<L; ++t) S_k[t] = Shapelets[k + t*K];

        for (int t = 0; t < L; ++t) {
            double s_val = S_k[t];
            double *p_grad_base = Grad + k + t * KN;
            
            // Term A
            for (int n = 0; n < N; ++n) {
                p_grad_base[n*K] = Sum_W[k + n*K] * s_val; 
            }

            // Term B
            for (int i = 0; i < Q; ++i) {
                double *p_w = W_all + k + i * KN;
                double *p_eeg = EEG + (i + t) * N;
                for (int n = 0; n < N; ++n) {
                    p_grad_base[n*K] -= p_w[n*K] * p_eeg[n];
                }
            }

            // Coeff
            double factor = 2.0 / (double)L;
            for (int n = 0; n < N; ++n) {
                p_grad_base[n*K] *= factor * Part1[k + n*K];
            }
        }
    }
}