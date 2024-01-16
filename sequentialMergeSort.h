template <class T>
static int Min(T a, T b) {
	return a <= b ? a : b;
}


template <class T>
static void sequentialMerge(T* data, int left, int mid, int right) {
	int i, j, k;
	int n1 = mid - left + 1;
	int n2 = right - mid;
	T* L = new T[n1];
	T* R = new T[n2];

	for (i = 0; i < n1; i++)
		L[i] = data[left + i];

	for (j = 0; j < n2; j++)
		R[j] = data[mid + 1 + j];

	i = 0;
	j = 0;
	k = left;

	while (i < n1 && j < n2)
	{
		if (L[i] >= R[j])
		{
			data[k] = L[i];
			i++;
		}
		else
		{
			data[k] = R[j];
			j++;
		}

		k++;
	}

	while (i < n1)
	{
		data[k] = L[i];
		i++;
		k++;
	}

	while (j < n2)
	{
		data[k] = R[j];
		j++;
		k++;
	}

	delete L;
	delete R;
}

template <class T>
static void sequentialMergeSort(T* data, int count) {
	int currentSize;
	int leftStart;

	for (currentSize = 1; currentSize <= count - 1; currentSize = 2 * currentSize)
	{
		for (leftStart = 0; leftStart < count - 1; leftStart += 2 * currentSize)
		{
			int mid = leftStart + currentSize - 1;
			int rightEnd = Min(leftStart + 2 * currentSize - 1, count - 1);

			sequentialMerge(data, leftStart, mid, rightEnd);
		}
	}
}