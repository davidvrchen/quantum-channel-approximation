import time


def time_script(func, *args, **kwargs):
    """Function to time execution of main function of scripts."""
    
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        total_time = end_time - start_time

        print(f"Executing script took {total_time:.4f} seconds")
        return result

    return timeit_wrapper
