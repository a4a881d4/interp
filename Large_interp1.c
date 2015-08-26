/***********************************************************
Company Name:
	�廪��ѧ��������;
Function Name:
	Large_interp;
Function Description:
	;���������ڲ�
Inputs:
	d_os_sig					   �� �����ź�
	os_sig_len                     �� �����źų���
	d_new_idx					   :  �ڲ�ϵ��
	new_idx_len                    �� �ڲ�ϵ������
	d_factor С��1�ϲ���,����1�²���
	d_out_sig					   :  ����ź�
	out_sig_len					   �� ����źų���
Outputs:
	����0��������
	����1�������źų��ȳ����趨���ֵ��
	����2������new_idx�ĳ��ȳ����趨���ֵ��
	����3���±�ֵ���������źų���

Notes: 
**************************************************************************/
#include "Large_interp1.h"

int Large_interp1(double d_os_sig_real[], double d_os_sig_imag[], int os_sig_len, Air_mid_la_para *p_Air_mid_la_para, double *d_out_sig_real, double *d_out_sig_imag, int *out_sig_len, double *d_out_fracst, int tx_rx_flag)
{
	int i = 0;
	int i_num;
	int i_num_tmp;
	int i_resample_len;
	int new_idx_len;

	double d_factor;
	double d_frac;
	double d_params = 1.0/3;
	double d_c0;
	double d_tmp1;
	double d_tmp2;
	double d_c1;
	double d_c2;
	double d_c3;
	double d_mult1;
	double d_mult2;
	double d_mult3;
	double d_new_idx[LARGE_INTERP_NEW_IDX_LEN];

	if (os_sig_len > LARGE_INTERP_OS_SIG_LEN)
	{
		return 1;
	}

//	if (new_idx_len > LARGE_INTERP_NEW_IDX_LEN)
//	{
//		return 2;
//	}

	////////////////// added by GAO 20150802
	//Bug
	i_resample_len = floor(os_sig_len*p_Air_mid_la_para->interp_factor);
	if (tx_rx_flag) d_factor = 1/p_Air_mid_la_para->interp_factor;
	else d_factor = p_Air_mid_la_para->interp_factor;

	for (i=0; i<i_resample_len; i++)
	{
		
		d_new_idx[i]= p_Air_mid_la_para->fracst + (p_Air_mid_la_para->timeo + i + 1)*d_factor;
		if (d_new_idx[i] > (os_sig_len-1))
		{
			new_idx_len = i;
			*d_out_fracst = d_new_idx[i-1] - os_sig_len + 4;
			break;
		}

	}
	
	printf("new_idx_len %d\n",new_idx_len);
	///////////////////////////////////////

	for (i=0; i<new_idx_len; i++)
	{
		i_num_tmp = floor(d_new_idx[i]);
		d_frac = d_new_idx[i] - i_num_tmp;
		i_num = i_num_tmp - 1;

		if (i_num > os_sig_len)
		{
			return 3;
		}

		d_c0 = d_os_sig_real[i_num];
		d_tmp1 = d_params * d_os_sig_real[i_num - 1];
		d_tmp2 = d_params * d_os_sig_real[i_num + 2];
		d_c1 = d_os_sig_real[i_num + 1] - d_tmp1 - d_os_sig_real[i_num]/2 - d_tmp2/2;
		d_c2 = (d_os_sig_real[i_num - 1] + d_os_sig_real[i_num + 1])/2 - d_os_sig_real[i_num];
		d_c3 = d_params/2 * (d_os_sig_real[i_num + 2] - d_os_sig_real[i_num - 1]) + (d_os_sig_real[i_num] - d_os_sig_real[i_num + 1])/2;	
		d_mult1 = d_c3 * d_frac;
		d_mult2 = (d_mult1 + d_c2) * d_frac;
		d_mult3 = (d_mult2 + d_c1) * d_frac;
		d_out_sig_real[i] = d_mult3 + d_c0;

		d_c0 = d_os_sig_imag[i_num];
		d_tmp1 = d_params * d_os_sig_imag[i_num - 1];
		d_tmp2 = d_params * d_os_sig_imag[i_num + 2];
		d_c1 = d_os_sig_imag[i_num + 1] - d_tmp1 - d_os_sig_imag[i_num]/2 - d_tmp2/2;
		d_c2 = (d_os_sig_imag[i_num - 1] + d_os_sig_imag[i_num + 1])/2 - d_os_sig_imag[i_num];
		d_c3 = d_params/2 * (d_os_sig_imag[i_num + 2] - d_os_sig_imag[i_num - 1]) + (d_os_sig_imag[i_num] - d_os_sig_imag[i_num + 1])/2;
		d_mult1 = d_c3 * d_frac;
		d_mult2 = (d_mult1 + d_c2) * d_frac;
		d_mult3 = (d_mult2 + d_c1) * d_frac;
		d_out_sig_imag[i] = d_mult3 + d_c0;
	}

	*out_sig_len = new_idx_len;

	return 0;
}
