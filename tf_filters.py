from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

def normalize_tuple(value, n, name):
    """Transforms an integer or iterable of integers into an integer tuple.
    Ported from tensorflow.python.keras.util.
    Arguments:
      value: The value to validate and convert. Could an int, or any iterable
        of ints.
      n: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.
    Returns:
      A tuple of n integers.
    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise TypeError('The `' + name + '` argument must be a tuple of ' +
                            str(n) + ' integers. Received: ' + str(value))
        if len(value_tuple) != n:
            raise ValueError('The `' + name +
                             '` argument must be a tuple of ' + str(n) +
                             ' integers. Received: ' + str(value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError('The `' + name +
                                 '` argument must be a tuple of ' + str(n) +
                                 ' integers. Received: ' + str(value) + ' '
                                 'including element ' + str(single_value) +
                                 ' of type' + ' ' + str(type(single_value)))
        return value_tuple


def _hasattr(obj, attr_name):
    # If possible, avoid retrieving the attribute as the object might run some
    # lazy computation in it.
    if attr_name in dir(obj):
        return True
    try:
        getattr(obj, attr_name)
    except AttributeError:
        return False
    else:
        return True


def assert_like_rnncell(cell_name, cell):
    """Raises a TypeError if cell is not like a
    tf.keras.layers.AbstractRNNCell.
    Args:
      cell_name: A string to give a meaningful error referencing to the name
        of the function argument.
      cell: The object which should behave like a
        tf.keras.layers.AbstractRNNCell.
    Raises:
      TypeError: A human-friendly exception.
    """
    conditions = [
        _hasattr(cell, "output_size"),
        _hasattr(cell, "state_size"),
        _hasattr(cell, "get_initial_state"),
        callable(cell),
    ]

    errors = [
        "'output_size' property is missing",
        "'state_size' property is missing",
        "'get_initial_state' method is required", "is not callable"
    ]

    if not all(conditions):
        errors = [error for error, cond in zip(errors, conditions) if not cond]
        raise TypeError("The argument {!r} ({}) is not an RNNCell: {}.".format(
            cell_name, cell, ", ".join(errors)))

def get_ndims(image):
    return image.get_shape().ndims or tf.rank(image)


@tf.function
def to_4D_image(image):
    """Convert 2/3/4D image to 4D image.
    Args:
      image: 2/3/4D tensor.
    Returns:
      4D tensor with the same type.
    """
    with tf.control_dependencies([
            tf.debugging.assert_rank_in(  # pylint: disable=bad-continuation
                image, [2, 3, 4],
                message='`image` must be 2/3/4D tensor')
    ]):
        ndims = image.get_shape().ndims
        if ndims is None:
            return _dynamic_to_4D_image(image)
        elif ndims == 2:
            return image[None, :, :, None]
        elif ndims == 3:
            return image[None, :, :, :]
        else:
            return image


def _dynamic_to_4D_image(image):
    shape = tf.shape(image)
    original_rank = tf.rank(image)
    # 4D image => [N, H, W, C] or [N, C, H, W]
    # 3D image => [1, H, W, C] or [1, C, H, W]
    # 2D image => [1, H, W, 1]
    left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    # yapf: disable
    new_shape = tf.concat(
        [tf.ones(shape=left_pad, dtype=tf.int32),
         shape,
         tf.ones(shape=right_pad, dtype=tf.int32)],
        axis=0)
    # yapf: enable
    return tf.reshape(image, new_shape)


@tf.function
def from_4D_image(image, ndims):
    """Convert back to an image with `ndims` rank.
    Args:
      image: 4D tensor.
      ndims: The original rank of the image.
    Returns:
      `ndims`-D tensor with the same type.
    """
    with tf.control_dependencies([
            tf.debugging.assert_rank(  # pylint: disable=bad-continuation
                image,
                4,
                message='`image` must be 4D tensor')
    ]):
        if isinstance(ndims, tf.Tensor):
            return _dynamic_from_4D_image(image, ndims)
        elif ndims == 2:
            return tf.squeeze(image, [0, 3])
        elif ndims == 3:
            return tf.squeeze(image, [0])
        else:
            return image


def _dynamic_from_4D_image(image, original_rank):
    shape = tf.shape(image)
    # 4D image <= [N, H, W, C] or [N, C, H, W]
    # 3D image <= [1, H, W, C] or [1, C, H, W]
    # 2D image <= [1, H, W, 1]
    begin = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
    end = 4 - tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
    new_shape = shape[begin:end]
    return tf.reshape(image, new_shape)

def _pad(image, filter_shape, mode="CONSTANT", constant_values=0):
    """Explicitly pad a 4-D image.
    Equivalent to the implicit padding method offered in `tf.nn.conv2d` and
    `tf.nn.depthwise_conv2d`, but supports non-zero, reflect and symmetric
    padding mode. For the even-sized filter, it pads one more value to the
    right or the bottom side.
    Args:
      image: A 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: A `tuple`/`list` of 2 integers, specifying the height
        and width of the 2-D filter.
      mode: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
    """
    assert mode in ["CONSTANT", "REFLECT", "SYMMETRIC"]
    filter_height, filter_width = filter_shape
    pad_top = (filter_height - 1) // 2
    pad_bottom = filter_height - 1 - pad_top
    pad_left = (filter_width - 1) // 2
    pad_right = filter_width - 1 - pad_left
    paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    return tf.pad(image, paddings, mode=mode, constant_values=constant_values)


@tf.function
def mean_filter2d(image,
                  filter_shape=(3, 3),
                  padding="REFLECT",
                  constant_values=0,
                  name=None):
    """Perform mean filtering on image(s).
    Args:
      image: Either a 2-D `Tensor` of shape `[height, width]`,
        a 3-D `Tensor` of shape `[height, width, channels]`,
        or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: An `integer` or `tuple`/`list` of 2 integers, specifying
        the height and width of the 2-D mean filter. Can be a single integer
        to specify the same value for all spatial dimensions.
      padding: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
      name: A name for this operation (optional).
    Returns:
      3-D or 4-D `Tensor` of the same dtype as input.
    Raises:
      ValueError: If `image` is not 2, 3 or 4-dimensional,
        if `padding` is other than "REFLECT", "CONSTANT" or "SYMMETRIC",
        or if `filter_shape` is invalid.
    """
    with tf.name_scope(name or "mean_filter2d"):
        image = tf.convert_to_tensor(image, name="image")
        original_ndims = get_ndims(image)
        image = to_4D_image(image)

        if padding not in ["REFLECT", "CONSTANT", "SYMMETRIC"]:
            raise ValueError(
                "padding should be one of \"REFLECT\", \"CONSTANT\", or "
                "\"SYMMETRIC\".")

        filter_shape = normalize_tuple(filter_shape, 2,
                                                   "filter_shape")

        # Keep the precision if it's float;
        # otherwise, convert to float32 for computing.
        orig_dtype = image.dtype
        if not image.dtype.is_floating:
            image = tf.dtypes.cast(image, tf.dtypes.float32)

        # Explicitly pad the image
        image = _pad(
            image, filter_shape, mode=padding, constant_values=constant_values)

        # Filter of shape (filter_width, filter_height, in_channels, 1)
        # has the value of 1 for each element.
        area = tf.constant(
            filter_shape[0] * filter_shape[1], dtype=image.dtype)
        filter_shape += (tf.shape(image)[-1], 1)
        kernel = tf.ones(shape=filter_shape, dtype=image.dtype)

        output = tf.nn.depthwise_conv2d(
            image, kernel, strides=(1, 1, 1, 1), padding="VALID")

        output /= area

        output = from_4D_image(output, original_ndims)
        return tf.dtypes.cast(output, orig_dtype)


@tf.function
def median_filter2d(image,
                    filter_shape=(3, 3),
                    padding="REFLECT",
                    constant_values=0,
                    name=None):
    """Perform median filtering on image(s).
    Args:
      image: Either a 2-D `Tensor` of shape `[height, width]`,
        a 3-D `Tensor` of shape `[height, width, channels]`,
        or a 4-D `Tensor` of shape `[batch_size, height, width, channels]`.
      filter_shape: An `integer` or `tuple`/`list` of 2 integers, specifying
        the height and width of the 2-D median filter. Can be a single integer
        to specify the same value for all spatial dimensions.
      padding: A `string`, one of "REFLECT", "CONSTANT", or "SYMMETRIC".
        The type of padding algorithm to use, which is compatible with
        `mode` argument in `tf.pad`. For more details, please refer to
        https://www.tensorflow.org/api_docs/python/tf/pad.
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode.
      name: A name for this operation (optional).
    Returns:
      3-D or 4-D `Tensor` of the same dtype as input.
    Raises:
      ValueError: If `image` is not 2, 3 or 4-dimensional,
        if `padding` is other than "REFLECT", "CONSTANT" or "SYMMETRIC",
        or if `filter_shape` is invalid.
    """
    with tf.name_scope(name or "median_filter2d"):
        image = tf.convert_to_tensor(image, name="image")
        original_ndims = get_ndims(image)
        image = to_4D_image(image)

        if padding not in ["REFLECT", "CONSTANT", "SYMMETRIC"]:
            raise ValueError(
                "padding should be one of \"REFLECT\", \"CONSTANT\", or "
                "\"SYMMETRIC\".")

        filter_shape = normalize_tuple(filter_shape, 2,
                                                   "filter_shape")

        image_shape = tf.shape(image)
        batch_size = image_shape[0]
        height = image_shape[1]
        width = image_shape[2]
        channels = image_shape[3]

        # Explicitly pad the image
        image = _pad(
            image, filter_shape, mode=padding, constant_values=constant_values)

        area = filter_shape[0] * filter_shape[1]

        floor = (area + 1) // 2
        ceil = area // 2 + 1

        patches = tf.image.extract_patches(
            image,
            sizes=[1, filter_shape[0], filter_shape[1], 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID")

        patches = tf.reshape(
            patches, shape=[batch_size, height, width, area, channels])

        patches = tf.transpose(patches, [0, 1, 2, 4, 3])

        # Note the returned median is casted back to the original type
        # Take [5, 6, 7, 8] for example, the median is (6 + 7) / 2 = 3.5
        # It turns out to be int(6.5) = 6 if the original type is int
        top = tf.nn.top_k(patches, k=ceil).values
        if area % 2 == 1:
            median = top[:, :, :, :, floor - 1]
        else:
            median = (
                top[:, :, :, :, floor - 1] + top[:, :, :, :, ceil - 1]) / 2

        output = tf.cast(median, image.dtype)
        output = from_4D_image(output, original_ndims)
        return output