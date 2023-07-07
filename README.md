<div align="center">

## mlcircuit

<h3>



</h3>

[![Unit Tests](https://github.com/Charbel199/mlcircuit/actions/workflows/test.yml/badge.svg)](https://github.com/Charbel199/mlcircuit/blob/main/.github/workflows/test.yml)

</div>



This project aims to provide a seamless integration of Kafka, ONNX, TensorRT, and PyTorch for end-to-end machine learning training, evaluation, and deployment. By leveraging these technologies, we can build a robust and efficient pipeline for developing and deploying machine learning models.

In reality, the true purpose of this project is to get better at these technologies while working on something fun.


## Features

- **Kafka Integration**: The project utilizes Apache Kafka, a distributed streaming platform, to enable real-time data ingestion and processing for machine learning tasks.

- **ONNX Support**: The Open Neural Network Exchange (ONNX) format is used as an intermediate representation of machine learning models. This allows for model interoperability and facilitates seamless integration with other frameworks and tools.

- **TensorRT Integration**: TensorRT is a high-performance deep learning inference optimizer and runtime library. It optimizes and deploys trained models onto GPUs, providing accelerated inferencing capabilities for our machine learning models.

- **PyTorch Integration**: PyTorch, a popular deep learning framework, is used for model training, evaluation, and deployment. It provides a flexible and intuitive interface for building and training neural networks.


### Kafka

```
cd mlcircuit/streaming/docker
docker-compose up
```


## License

This project is licensed under the MIT License. See the LICENSE file for details.