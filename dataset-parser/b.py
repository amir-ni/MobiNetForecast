import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

for name in ["Rome", "Porto", "GeoLife"]:
    for res in ["6","7","8","9"]:
        # Load data and preprocess
        a = pd.read_csv("/local/data1/shared_data/higher_order_trajectory/"+name.lower()+"/ho_"+name.lower()+"_res"+res+".csv")["higher_order_trajectory"].to_list()
        a = [x.split() for x in a]  # Split each trajectory into individual items

        a = [item for sublist in a for item in sublist]  # Flatten the list of lists

        # Step 1: Calculate the frequency of each item
        distribution = {}
        for item in a:
            if item in distribution:
                distribution[item] += 1
            else:
                distribution[item] = 1

        # Step 3: Sort the distribution of frequencies
        sorted_frequency_counts = dict(sorted(distribution.items(), key=lambda x: -x[1]))

        # Step 4: Plot the distribution of frequencies
        plt.figure(figsize=(12, 8))

        # Bar plot with improved visualization
        plt.bar(
            sorted_frequency_counts.keys(), 
            sorted_frequency_counts.values(), 
            color='skyblue', 
            edgecolor='blue', 
            width=0.8
        )

        plt.xlabel('Hexagons', fontsize=14)
        plt.ylabel('Number of Traversal (Log Scale)', fontsize=14)
        plt.title('Number of Traversal for Each Hexagon ('+name+' '+res+')', fontsize=18)

        # Set a logarithmic scale for the y-axis to handle large variances better
        plt.yscale('log')
        maxi = list(sorted_frequency_counts.keys())[-1]
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot
        file_path = '/home/anadiri/collision/distributions/hex-' +name.lower()+res+'.png'
        plt.savefig(file_path, dpi=300)  # Increased DPI for higher resolution

        # Optionally, display the plot
        plt.show()
