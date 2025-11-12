#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int classLabel;
    int* features;
} Sample;

void naive_bayes_train(Sample* dataset, int numSamples, int numFeatures, 
                       int numClasses, int numFeatureValues,
                       int* priors, int* likelihoods) {
    for (int i = 0; i < numSamples; i++) {
        int classLabel = dataset[i].classLabel;
        priors[classLabel]++;
        
        for (int f = 0; f < numFeatures; f++) {
            int featureValue = dataset[i].features[f];
            int likelihoodIndex = classLabel * numFeatures * numFeatureValues + 
                                 f * numFeatureValues + featureValue;
            likelihoods[likelihoodIndex]++;
        }
    }
}

int main() {
    const int numSamples = 1000;
    const int numFeatures = 10;
    const int numClasses = 3;
    const int numFeatureValues = 5;

    Sample* dataset = (Sample*)malloc(numSamples * sizeof(Sample));
    for (int i = 0; i < numSamples; i++) {
        dataset[i].features = (int*)malloc(numFeatures * sizeof(int));
        dataset[i].classLabel = rand() % numClasses;
        for (int f = 0; f < numFeatures; f++) {
            dataset[i].features[f] = rand() % numFeatureValues;
        }
    }

    int* priors = (int*)calloc(numClasses, sizeof(int));
    int* likelihoods = (int*)calloc(numClasses * numFeatures * numFeatureValues, sizeof(int));

    naive_bayes_train(dataset, numSamples, numFeatures, numClasses, 
                      numFeatureValues, priors, likelihoods);

    printf("Naive Bayes training completed\n");

    for (int i = 0; i < numSamples; i++) {
        free(dataset[i].features);
    }
    free(dataset);
    free(priors);
    free(likelihoods);

    return 0;
}
