from typing import Any, List
from encoder.embedding import Embeddings
import umap
import hdbscan
import numpy as np
import heapq

class Searching_results:
    def __init__(self, threshold:float=0.01):
        self.threshold = threshold
        self.reducer = umap.UMAP(
            n_components=3,
            n_neighbors=2,
            min_dist=0.01,
            metric='cosine',
            n_epochs=100
        )
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
        self.encoder = Embeddings()

    def __call__(self, documents: List[str], embeddings:List[float], query: str, structured_result:bool=True) -> Any:
        

        reduced_embeddings = self.reducer.fit_transform(embeddings)
        clusters = self.clusterer.fit_predict(reduced_embeddings)

        # Process query
        encoded_query = self.encoder([query])
        reduced_query_embedding = self.reducer.transform(encoded_query)

        # # Step 3: Calculate centroids and similarity scores
        # unique_clusters, centroid_array = [], []

        # # Calculate centroids for unique clusters
        # for cluster in np.unique(clusters):
        #     indices = np.where(clusters == cluster)[0]
        #     cluster_embeddings = reduced_embeddings[indices]
        #     centroid = np.mean(cluster_embeddings, axis=0)
        #     unique_clusters.append(cluster)
        #     centroid_array.append(centroid)

        # # Convert centroid_array to NumPy array
        # centroid_array = np.array(centroid_array)

        # # Calculate similarity scores for centroids
        # sim_scores_centroid = np.array([self.get_similarity_score(reduced_query_embedding, centroid) for centroid in centroid_array])



        unique_clusters= list(set(clusters))
        
        centroid_array = list([])
        for element in unique_clusters:
            indices = reduced_embeddings[np.where(clusters == element)]
            centroid = np.mean(indices, axis=0)
            centroid_array.append(centroid)

        sim_scores_centroid = list([])
        for doc in centroid_array:
            sim_score = self.get_similarity_score(reduced_query_embedding, doc)
            sim_scores_centroid.append(sim_score)
    
        
        max_data = np.max(sim_scores_centroid) 
        data_threshold = max_data - self.threshold 
        
        filter_data = [unique_clusters[i] for i, data in enumerate(sim_scores_centroid) if data >= data_threshold]
        
        filter_results = list([])
        filter_results_embeddings = list([])
        filter_clusters = list([])

        for filter in filter_data:
            index = np.where(clusters == filter)[0]
            filter_results.extend(np.asarray(documents)[index])
            filter_results_embeddings.extend(reduced_embeddings[index])
            filter_clusters.append(filter)
        
        sim_scores_filtered = list([])
        for doc in filter_results_embeddings:
            sim_score = self.get_similarity_score(reduced_query_embedding, doc)
            sim_scores_filtered.append(sim_score) 

        results = [i for i in zip(sim_scores_filtered, filter_results, filter_clusters)]
        results.sort(key=lambda x: ( x[0]), reverse=True)

        if structured_result:
            results_structure = {
            
                "search_results":results,
                "reduced_embeddings": reduced_embeddings,
                "reduced_query_embedding": reduced_query_embedding,
                "clusters": clusters
            }

            return(results_structure)
        return results


    @classmethod
    def filter_repeated_text(self, documents:List[str])-> List[str]:
        """
            Filters out repeated text from the input list.

            Parameters:
            input_list (list of str): The list of strings to filter.

            Returns:
            list of str: A list with only unique strings, preserving the order of first occurrence.
        """
        seen = set()
        documents_list = []
    
        for item in documents:
            if item not in seen:
                seen.add(item)
                documents_list.append(item)
    
        return documents_list
    
    @classmethod 
    def get_similarity_score(self, vestor_a:List[float], vector_b:List[float])-> float:
        sim_score = np.dot(vestor_a, vector_b) / (
                np.linalg.norm(vestor_a) * np.linalg.norm(vector_b) + 1e-10
            )
        return sim_score
    
    @classmethod
    def plot_search(self, results_structure:Any)->None:
        try:
            import plotly.graph_objs as go
        except ImportError:
            print("Plotting is disabled. Please `pip install plotly`.")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Plotting is disabled. Please `pip install matplotlib`.")
            return


        # Assuming reduced_embeddings, clusters, and reduced_query_embedding are defined

        # Convert the reduced embeddings to a numpy array for easier indexing
        reduced_embeddings = np.array(results_structure["reduced_embeddings"])
        
        dimension = reduced_embeddings.shape[1]
        if dimension != 3:
            print(f"Data need to have 3D and it has {dimension} dimensions")
            return
       
        reduced_query_embedding = results_structure["reduced_query_embedding"][0]

        # Extract x, y, z coordinates for each candidate
        # x = reduced_embeddings[:, 0]
        # y = reduced_embeddings[:, 1]
        # z = reduced_embeddings[:, 2]

        # Plot each cluster with a different color
        unique_clusters = set(results_structure["clusters"])
        cmap = plt.cm.get_cmap('tab10')  # You can choose different colormaps here
        colors = [cmap(i) for i in np.linspace(0, 1, len(unique_clusters))]

# Convert RGBA tuple to rgba format for CSS-style color strings
        colors = [f'rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 0.8)' for c in colors]

        data = []

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                # Noise points
                color = 'black'
                marker = 'x'
            else:
                color = colors[cluster_id]
                marker = 'circle'
            cluster_points = reduced_embeddings[results_structure["clusters"] == cluster_id]
            trace = go.Scatter3d(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                z=cluster_points[:, 2],
                mode='markers',
                marker=dict(size=5, color=color, symbol=marker),
                name=f'Cluster {cluster_id}'
            )
            data.append(trace)

        # Highlight the query embedding
        query_trace = go.Scatter3d(
            x=[reduced_query_embedding[0]],
            y=[reduced_query_embedding[1]],
            z=[reduced_query_embedding[2]],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='Query'
        )
        data.append(query_trace)

        # Set plot labels and title
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='x'),
                yaxis=dict(title='y'),
                zaxis=dict(title='z')
            ),
            title='3D Cluster Plot of Candidates'
        )

        # Create the figure
        fig = go.Figure(data=data, layout=layout)

        # Show plot
        fig.show()

    

