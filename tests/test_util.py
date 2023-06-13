# %%
import numpy as np
from waveorder import util


def test_pad_array():
    # Test case 1
    input_array = np.ones((10, 3, 100, 100, 100))  # 5D array
    pad = 10
    pad_dims = (1, 4)
    expected_shape = (10, 23, 100, 100, 120)  # Expected padded shape
    padded_array = util.pad_array(input_array, pad, pad_dims)
    assert padded_array.shape == expected_shape

    # Test case 2
    input_array = np.zeros((5, 1, 50, 50, 50))  # 5D array
    pad = 5
    pad_dims = (2, 3)
    expected_shape = (5, 1, 60, 60, 50)  # Expected padded shape

    padded_array = util.pad_array(input_array, pad, pad_dims)
    assert padded_array.shape == expected_shape


def test_generate_FOV_splitting_parameters():
    # Test case 1: Default behavior, not enforcing power of two
    # img_size = (2048, 2048)
    # overlapping_range = (5, 50)
    # max_image_size = (300, 300)
    # overlap, N_space, M_space = util.generate_FOV_splitting_parameters(
    #     img_size, overlapping_range, max_image_size
    # )
    # print(overlap, N_space, M_space)
    # assert overlap == 5
    # assert N_space == 227
    # assert M_space == 227

    # # Test case 2: Enforcing power of two
    img_size = (2048, 2048)
    overlapping_range = (2, 30)
    max_image_size = (512, 512)
    overlap, N_space, M_space = util.generate_FOV_splitting_parameters(
        img_size, overlapping_range, max_image_size, power_of_two=True
    )
    print(overlap, N_space, M_space)
    assert overlap == 12
    assert N_space == 4
    assert M_space == 4

    # # Test case 3: Image size is already a power of two
    img_size = (2048, 2048)
    overlapping_range = (2, 250)  # Adjusted overlapping range
    max_image_size = (512, 512)
    overlap, N_space, M_space = util.generate_FOV_splitting_parameters(
        img_size, overlapping_range, max_image_size, power_of_two=True
    )
    print(overlap, N_space, M_space)
    assert overlap == 128
    assert N_space == 384
    assert M_space == 384
    print("All tests passed successfully.")



def best_case_FOV_splitting():
    best_overlap = float('inf')
    best_N_space = 0
    best_M_space = 0

    for img_size in range(2048, 2500):
        for overlap in range(5, 100):
            for max_image_size in range(250, 600):
                try:
                    overlap_val, N_space_val, M_space_val = util.generate_FOV_splitting_parameters(
                        (img_size, img_size),
                        (overlap, overlap + 1),
                        (max_image_size, max_image_size),
                        power_of_two=True
                    )

                    if overlap_val < best_overlap:
                        best_img_size = img_size
                        best_overlap = overlap
                        best_max_image_size = max_image_size
                        best_overlap = overlap_val
                        best_N_space = N_space_val
                        best_M_space = M_space_val

                except ValueError:
                    continue
    print("Best best_img_size:", best_img_size)
    print("Best best_max_image_size:", best_max_image_size)
    print("Best overlap:", best_overlap)
    print("Best overlap:", best_overlap)
    print("Best N_space:", best_N_space)
    print("Best M_space:", best_M_space)

# Run the test cases
if __name__ == "__main__":
    # Run the tests
    test_pad_array()
    # test_generate_FOV_splitting_parameters()
    # best_case_FOV_splitting()

