__kernel void multiply(__global int* arr, int a,__global int* sum)
{
	int i = get_global_id(0);

	arr[i] = arr[i] *a;
	//sum[0] += arr[i];
	atomic_add(sum, arr[i]);
}
