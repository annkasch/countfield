# countfield

Learn a continuous field p(x) from sparse count data: at each point x_i, k_i successes
out of N_i trials, k_i ~ Binomial(N_i, p(x_i)). A set-to-point model (neural process over
points) predicts p and a predictive count distribution at query points from the observed
points around them, trained by masking observed points.

Application-independent. First application: the LEGEND LAr optical map, with data from
[legend-optical-map-ml](https://github.com/annkasch/legend-optical-map-ml).

Status: skeleton.
