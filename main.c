#include <stdio.h>
#include <stdlib.h>

//QuickSelect 将第k小的元素放在 a[k-1]
void QuickSelect(int a[], int k, int left, int right)
{
    int i, j;
    int pivot;
    int cutoff;

    if (left + cutoff <= right)
    {
        pivot = median3(a, left, right);
        //取三数中值作为枢纽元，可以很大程度上避免最坏情况
        i = left;
        j = right - 1;
        for (;;)
        {
            while (a[++i] < pivot)
            {
            }
            while (a[--j] > pivot)
            {
            }
            if (i < j)
                swap(&a[i], &a[j]);
            else
                break;
        }
        //重置枢纽元
        swap(&a[i], &a[right - 1]);

        if (k <= i)
            QuickSelect(a, k, left, i - 1);
        else if (k > i + 1)
            QuickSelect(a, k, i + 1, right);
    }
    else
        InsertSort(a + left, right - left + 1);
}

int main(void) {

    return 0;
}