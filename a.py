import pandas as pd

# Replace with the actual path to your CSV file
file_path = "/data/jongseo/project/Video-CBM-two-stream/kinetics400_resize_test.csv"

# Load the CSV file
df = pd.read_csv(file_path, header=None, names=['filename', 'label'])

# Add the path to the beginning of each filename
df['filename'] = '/local_datasets/kinetics400_320p/test/' + df['filename']

# Display the updated DataFrame (optional, for checking)
print(df.head())

# Save the modified DataFrame back to a new CSV file
output_file_path = "./test.csv"  # Replace with your desired output file path
df.to_csv(output_file_path, index=False)