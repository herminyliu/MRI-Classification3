def apply_slice(slice_obj, input_list):
    """
    Apply a slice to a list and return the result.

    Arguments:
    slice_obj -- A slice object specifying the range of elements to be selected.
    input_list -- The list to which the slice will be applied.

    Returns:
    The result of applying the slice to the input list.
    """
    return input_list[slice_obj]


# 示例用法
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
my_slice = slice(2, 7)  # 选择索引2到6之间的元素
result = apply_slice(my_slice, my_list)
print(result)
