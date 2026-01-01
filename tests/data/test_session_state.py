# Test session state persistence

# First run: initialize x if it doesn't exist
if "x" not in globals():
    x = 0
    print("Initializing x to 0")
else:
    print(f"Found existing x = {x}")

# Increment x
x += 1
print(f"x is now {x}")

# Store the result in _ to make it the return value
_ = x
