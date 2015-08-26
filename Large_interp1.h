#include <stdio.h>
#include <string.h>
#include <math.h>

/*拉格朗日内插*/

#define LARGE_INTERP_OS_SIG_LEN 1102000     /*输入信号长度最大值*/
#define LARGE_INTERP_NEW_IDX_LEN 580000    /*内插系数长度最大值*/

typedef struct tag_Air_mid_la
{
	double		interp_factor;
	double		fracst;
	double		timeo; 
}Air_mid_la_para;	


//int Large_interp(double d_os_sig[], int os_sig_len, double d_new_idx[], int new_idx_len, double *d_out_sig, int *out_sig_len);
int Large_interp1(double d_os_sig_real[], double d_os_sig_imag[], int os_sig_len, Air_mid_la_para *p_Air_mid_la_para, double *d_out_sig_real, double *d_out_sig_imag, int *out_sig_len, double *d_out_fracst, int tx_rx_flag);


