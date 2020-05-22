import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    
    centers = []
    first_choice = generator.choice(n, 1)
    centers.append(first_choice[0])
    for i in range(n_cluster - 1):
#        print(i)
        all_distances = np.array([min([np.linalg.norm(xx-x[c])**2 for c in centers]) for xx in x])
#        print(all_dists)
        
        probs = np.where(all_distances, all_distances/np.sum(all_distances), 0)
       
        new_centre  = np.argmax(probs)
#        print(new_centre)
        centers.append(new_centre)
#        print(centers)
    
    
#    print(centers)
    
    
#    raise Exception(
#             'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')

    

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers

    

def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator
        

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
#        print("Self Details:")
#        print("Clusters:", self.n_cluster)
#        print("Max Iter:", self.max_iter)
#        print("E:", self.e)
#        print("Data Shape:", x.shape)
        centroids = []
        print("Centres:", self.centers)
        for i in range(self.n_cluster):
            centroids.append(x[self.centers[i]])
        
        centroids = np.array(centroids)
        distort_factor = 10**10
        iterations = 0
#        for c in range(len(centroids)):
#            self.classification[c] = []
#        print("Centroids:", centroids)
        while iterations < self.max_iter:
            print(iterations)
#            for c in range(len(centroids)):
#                self.classification[c] = []
            all_dists = np.array([np.linalg.norm(x - centroids[centroid_centre], axis = 1)**2 for centroid_centre in range(len(centroids))])
#            print("All Distances Calculated.")
#            print(all_dists)
#            for centroid_centre in range(len(centroids)):
#                assign = np.sum((x-centroids[centroid_centre])**2, axis = 1)**.5
#                all_dists.append(assign)
            y_cent = np.array(np.argmin(all_dists, axis = 0))
#            for new_index in range(len(y)):
#                self.classification[y[new_index]].append(x[new_index])
#            print("All Centroids Updated.")
#            print("Y:", y)
#            for data_point in range(N):
#                dists = []
#                for centroid_centre in range(len(centroids)):
#                    current_cluster = centroids[centroid_centre]
##                    print("Cluster:", current_cluster)
##                    print("Data Point:", x[data_point])
##                    print("Number of Data Points:", N)
#                    euclid_dist = self.euclidean_distance(x[data_point], current_cluster)
#                    dists.append(euclid_dist)
##                    print("Distances:", dists)
#                y[data_point] = np.argmin(dists)
#            print("Y:",y)    
#            print("Y == Y-NP:", y==y_np)
            new_distortion_factor = 0
            
            for cluster in range(len(centroids)):
                clusters_vals = np.where(y_cent == cluster, 1, 0)
#                print(clusters_vals)
#                print([gg for gg in range(len(y)) if y[gg] == cluster])
#                print(all_points)
#                all_dists = np.sum((x-centroids[cluster])**2, axis = 1)
#                print(all_dists)
                new_distortion_factor += np.dot(all_dists[:][cluster], clusters_vals)
#            new_distortion_factor = np.sum(new_distortion_factor)
#            print(new_distortion_factor)
#            for point in range(N):
#                for cluster in range(len(centroids)):
#                    if y[point] == cluster:
#                        p_cluster = 1
#                    else:
#                        p_cluster = 0
#                    current_cluster = centroids[cluster]
#                    dist = self.euclidean_distance(x[point], current_cluster)
#                    new_distortion_factor += p_cluster*dist
#            print("ND:", new_distortion_factor)
#            print("Avg ND", new_distortion_factor/N)
            new_distortion_factor /= N
#            print("DIFF:", abs(distort_factor - new_distortion_factor))
            if abs(distort_factor - new_distortion_factor) <= self.e:
                break
            else:
                distort_factor = new_distortion_factor
                for centre in range(len(centroids)):
                    all_pts = np.where(y_cent == centre)[0]
                    all_labels = [x[gg] for gg in all_pts]
#                    pts = np.where(y == centre)
#                    print(pts)
#                    ptrs_2 = np.where(pts, x[pts], 0)
#                    print(ptrs_2)
#                    ptrs_2 = np.array(ptrs_2)
                    new_centre = np.average(all_labels, axis = 0)
                    centroids[centre] = new_centre
#                for centre in range(len(centroids)):
#                    all_points = []
#                    for a_cluster in range(len(y)):
#                        if y[a_cluster] == centre:
#                            all_points.append(x[a_cluster])
#                    if all_points != []:
##                        print("All Points:", all_points)
#                        all_points = np.array(all_points)
#                        new_centre = np.sum(all_points, axis = 0)/len(all_points)
#                        centroids[centre] = new_centre
#                
            iterations+= 1
         
#        iterations = i
#        print("Kmeans Fit Complete")
#        y = np.array(y)
        
#        raise Exception(
#             'Implement fit function in KMeans class')
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        print("Centroids:", centroids)
        print("Membership:", y_cent)
        return centroids, y_cent, iterations

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator
        self.classification = {}


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"
        
        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels
        

        # DONOT CHANGE CODE ABOVE THIS LINE
        
        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
#        print("Centers:", self.centers)
#        print("Y Train:", set(y))
        centroids = []
        y_centroids = [0]*N
#        print("Centres:", self.centers)
        for i in range(self.n_cluster):
            centroids.append(x[self.centers[i]])
        
        centroids = np.array(centroids)
        i = 0
        distort_factor = 10**10
        
#        for c in range(len(centroids)):
#            self.classification[c] = []
#        print("Centroids:", centroids)
        while i < self.max_iter:
#            for c in range(len(centroids)):
#                self.classification[c] = []
            all_dists = np.array([np.linalg.norm(x - centroids[centroid_centre], axis = 1) for centroid_centre in range(len(centroids))])
#            print(all_dists)
#            for centroid_centre in range(len(centroids)):
#                assign = np.sum((x-centroids[centroid_centre])**2, axis = 1)**.5
#                all_dists.append(assign)
            y_centroids = np.array(np.argmin(all_dists, axis = 0))
#            for new_index in range(len(y_centroids)):
#                self.classification[y_centroids[new_index]].append(x[new_index])
#            print("Y:", y)
#            for data_point in range(N):
#                dists = []
#                for centroid_centre in range(len(centroids)):
#                    current_cluster = centroids[centroid_centre]
##                    print("Cluster:", current_cluster)
##                    print("Data Point:", x[data_point])
##                    print("Number of Data Points:", N)
#                    euclid_dist = self.euclidean_distance(x[data_point], current_cluster)
#                    dists.append(euclid_dist)
##                    print("Distances:", dists)
#                y[data_point] = np.argmin(dists)
#            print("Y:",y)    
#            print("Y == Y-NP:", y==y_np)
            new_distortion_factor = 0
            
            for cluster in range(len(centroids)):
                clusters_vals = np.where(y_centroids == cluster, 1, 0)
#                print(clusters_vals)
#                print([gg for gg in range(len(y)) if y[gg] == cluster])
#                print(all_points)
#                all_dists = np.sum((x-centroids[cluster])**2, axis = 1)
#                print(all_dists)
                new_distortion_factor += np.dot(all_dists[:][cluster], clusters_vals)
#            new_distortion_factor = np.sum(new_distortion_factor)
#            print(new_distortion_factor)
#            for point in range(N):
#                for cluster in range(len(centroids)):
#                    if y[point] == cluster:
#                        p_cluster = 1
#                    else:
#                        p_cluster = 0
#                    current_cluster = centroids[cluster]
#                    dist = self.euclidean_distance(x[point], current_cluster)
#                    new_distortion_factor += p_cluster*dist
#            print("ND:", new_distortion_factor/N)
#            print("DIFF:", abs(distort_factor - new_distortion_factor))
            new_distortion_factor /= N
            if abs(distort_factor - new_distortion_factor) <= self.e:
                break
            else:
                distort_factor = new_distortion_factor
                for centre in range(len(centroids)):
                    all_pts = np.where(y_centroids == centre)[0]
                    all_labels = [x[gg] for gg in all_pts]
#                    pts = np.where(y == centre)
#                    print(pts)
#                    ptrs_2 = np.where(pts, x[pts], 0)
#                    print(ptrs_2)
#                    ptrs_2 = np.array(ptrs_2)
                    new_centre = np.average(all_labels, axis = 0)
                    centroids[centre] = new_centre
#                for centre in range(len(centroids)):
#                    all_points = []
#                    for a_cluster in range(len(y)):
#                        if y[a_cluster] == centre:
#                            all_points.append(x[a_cluster])
#                    if all_points != []:
##                        print("All Points:", all_points)
#                        all_points = np.array(all_points)
#                        new_centre = np.sum(all_points, axis = 0)/len(all_points)
#                        centroids[centre] = new_centre
#                
                i+= 1
        centroid_labels = []
        for i in range(len(centroids))  :
            all_i = np.where(y_centroids == i)[0]
            if len(all_i) == 0:
                centroid_labels.append(0)
            else:
#                print("ALL I", all_i, len(all_i))
                all_labels = [y[gg] for gg in all_i]
#                print(all_labels)
#                print(np.bincount(all_labels))
                centroid_labels.append(np.argmax(np.bincount(all_labels)))
            
            
            
          
        centroid_labels = np.array(centroid_labels)
#        self.centroid_labels = centroid_labels
         
        
        
#        raise Exception(
#             'Implement fit function in KMeansClassifier class')

        

        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        print("Centroid Labels:", self.centroid_labels)
        self.centroids = centroids
        print("Centroids:", centroids)
        print("Membership:", y_centroids)
        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        
        labels = [self.centroid_labels[np.argmin([np.linalg.norm(xx-cc) for cc in self.centroids])] for xx in x]
        
        
#        raise Exception(
#             'Implement predict function in KMeansClassifier class')
        
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
#    print("Images:",image[:5])
#    print("CVs:",code_vectors[:5])
#    new_im = []
#    i = 0
#    j = 0
#    for data_stream in image:
#        print("Data Stream " + str(i) + " in progress.")
#        for data_point in data_stream:
#            print("Data Point " + str(j) + " in progress.")
#            for centroid in code_vectors:
#                new_dist = np.argmin(np.linalg.norm(data_point-centroid))
#                new_im.append(code_vectors[new_dist])
#        i+= 1  
#        j+= 1
#    new_im = np.array(new_im)
#    print(image.shape)
#    print(new_im.shape)
#    print("Processing Image Reshape:")
#    new_im = [[[code_vectors[np.argmin(np.linalg.norm(data_point - centroid))] for centroid in code_vectors] for data_point in data_stream] for data_stream in image]
    X,Y,Z = image.shape
#    print(X, Y, Z)
    image = image.reshape(X*Y,Z)
#    print("Length of Image:", image.shape)
    new_im = [code_vectors[np.argmin([np.linalg.norm(data_point-centroid) for centroid in code_vectors])] for data_point in image]
    new_im = np.array(new_im)
    new_im = new_im.reshape(X,Y,Z)
#    print(new_im.shape)
#    raise Exception(
#             'Implement transform_image function')
    

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

