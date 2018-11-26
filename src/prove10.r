library(datasets)
library(cluster)

# Part 1: Agglomerative Heirarchical Clustering
# Step 1
myData = state.x77

#Step 2
# Compute the distance matrix of the unnormalized data
distance = dist(as.matrix(myData))

# Now perform the clustering of the unnormalized data
hc = hclust(distance)

# Plot the dendrogram of the unnormalized data
plot(hc)

# Step 3
# Normalize the data
myNormalizedData = scale(myData)

# Compute the distance matrix of the normalized data
distance = dist(as.matrix(myData))

# Perform the clustering of the unnormalized data
hc = hclust(distance)

# Plot the dendrogram of the unnormalized data
plot(hc)

# Step 4
# Remove 'area' from the attributes
myReducedData <- subset(myData, select = -c(Area))

# Normalize the reduced data
myReducedNormalizedData <- scale(myReducedData)

# Compute the distance matrix of the normalized data
distance = dist(as.matrix(myReducedNormalizedData))

# Perform the clustering of the unnormalized data
hc = hclust(distance)

# Plot the dendrogram of the unnormalized data
plot(hc)

# Step 5
# Get just the frost data
myFrostData <- subset(myData, select = c(Frost))

# Normalize the frost data
myNormalizedFrostData <- scale(myFrostData)

# Compute the distance matrix of the normalized data
distance = dist(as.matrix(myFrostData))

# Cluster on only the frost attribute
hc = hclust(distance)

# Plot the dendogram of the frost data
plot(hc)

# Part 2: Using K-Means

# Step 1
# Make sure to use a normalized version of the dataset.
myData <- state.x77
myNormalizedData <- scale(myData)

# Step 2
# Using k-means, cluster the data into 3 clusters. 
myClusters = kmeans(myNormalizedData, 3)

# Step 3
# Using a for loop, repeat the clustering process for k = 1 to 25, 
# plot the total within-cluster sum of squares error for each k-value.
sumOfSquaresError <- NULL;
numberOfRecords = length(myNormalizedData)
for (k in 1:25){
  print(k);
  myClusters = kmeans(myNormalizedData, k);
  sumOfSquaresError[k] <- myClusters$tot.withinss
}
plot(sumOfSquaresError)

# Step 4
# Evaluate the plot from the previous item, and choose an appropriate k-value using the "elbow method".
# Then re-cluster a single time using that k-value. Use this clustering for the remaining questions.
myClusters = kmeans(myNormalizedData, 4);

# Step 5
# List the states in each cluster.
print(myClusters$cluster)

# Step 6
# Use "clusplot" to plot a 2D representation of the clustering.
clusplot(myNormalizedData, myClusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)

# Step 7
# Analyze the centers of each of these clusters. Can you identify any insight into this clustering?
print(myClusters$centers)







