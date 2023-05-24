# Load the dataset
data(iris)

# Now, we will scale the feature
iris[,1:4] = scale(iris[,1:4])

iris$Species = as.numeric(iris$Species) -1

set.seed(123)
ind <- sample(2, nrow(iris), replace = TRUE, prob = c(0.7, 0.3))
trainData <- iris[ind == 1, 1:4]
testData <- iris[ind == 2, 1:4]
trainLabels <- to_categorical(iris[ind == 1, 5])
testLabels <- to_categorical(iris[ind == 2, 5])

# Building the model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 3, 
              activation ='softmax', 
              input_shape = ncol(trainData))

# Compile the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

# Convert dataframe into matrix 
trainData <- as.matrix(iris[ind == 1, 1:4])
testData <- as.matrix(iris[ind == 2, 1:4])

# Now we will train the model 
history <- model %>% fit(
  trainData,
  trainLabels,
  epochs = 1000,
  batch_size = 8,
  validation_split = 0.2
) 

score <- model %>% evaluate(testData, testLabels)
cat('Test loss:', score[1], '\n')
cat('Test accuracy:', score[2], '\n')


keras::save_model_hdf5(model, "my_model.h5")




  
