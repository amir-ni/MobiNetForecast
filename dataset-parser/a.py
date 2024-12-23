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

        # Step 2: Calculate the distribution of frequencies
        frequency_counts = {}
        for freq in distribution.values():
            if freq in frequency_counts:
                frequency_counts[freq] += 1
            else:
                frequency_counts[freq] = 1

        # Step 3: Sort the distribution of frequencies
        sorted_frequency_counts = dict(frequency_counts)

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

        plt.xlabel('Frequency of Traversal (Log Scale)', fontsize=14)
        plt.ylabel('Number of Hexagons (Log Scale)', fontsize=14)
        plt.title('Number of Hexagons per Frequency of Traversal ('+name+' '+res+')', fontsize=18)

        # Set a logarithmic scale for the y-axis to handle large variances better
        plt.xscale('log')
        plt.yscale('log')
        maxi = max(sorted_frequency_counts.keys())
        maxis = [int(maxi/4), int(2*maxi/4), int(3*maxi/4), maxi]
        maxis = [1] + [x//100*100 for x in maxis]
        # plt.xticks(ticks=maxis, labels=[str(x) for x in maxis])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot
        file_path = '/home/anadiri/collision/distributions/dist-' +name.lower()+res+'.png'
        plt.savefig(file_path, dpi=300)  # Increased DPI for higher resolution

        # Optionally, display the plot
        plt.show()
        # break
