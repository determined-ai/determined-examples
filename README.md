# Determined Legacy Examples

This Repository contains Determined examples that are no longer actively maintained by the determined team.

## Tutorials

|                          Example                           |    Dataset    |       Framework       |
| :--------------------------------------------------------: | :-----------: | :-------------------: |
| [fashion_mnist_tf_keras](tutorials/fashion_mnist_tf_keras) | Fashion MNIST | TensorFlow (tf.keras) |

## Computer Vision

|                                  Example                                   |           Dataset            |       Framework       |
| :------------------------------------------------------------------------: | :--------------------------: | :-------------------: |
|             [cifar10_pytorch](computer_vision/cifar10_pytorch)             |           CIFAR-10           |        PyTorch        |
|   [cifar10_pytorch_inference](computer_vision/cifar10_pytorch_inference)   |           CIFAR-10           |        PyTorch        |
|            [cifar10_tf_keras](computer_vision/cifar10_tf_keras)            |           CIFAR-10           | TensorFlow (tf.keras) |
|     [fasterrcnn_coco_pytorch](computer_vision/fasterrcnn_coco_pytorch)     |      Penn-Fudan Dataset      |        PyTorch        |
|         [mmdetection_pytorch](computer_vision/mmdetection_pytorch)         |             COCO             |        PyTorch        |
|           [detr_coco_pytorch](computer_vision/detr_coco_pytorch)           |             COCO             |        PyTorch        |
| [deformabledetr_coco_pytorch](computer_vision/deformabledetr_coco_pytorch) |             COCO             |        PyTorch        |
|               [iris_tf_keras](computer_vision/iris_tf_keras)               |         Iris Dataset         | TensorFlow (tf.keras) |
|              [unets_tf_keras](computer_vision/unets_tf_keras)              |   Oxford-IIIT Pet Dataset    | TensorFlow (tf.keras) |
|        [efficientdet_pytorch](computer_vision/efficientdet_pytorch)        |             COCO             |        PyTorch        |
|                [byol_pytorch](computer_vision/byol_pytorch)                | CIFAR-10 / STL-10 / ImageNet |        PyTorch        |
|    [deepspeed_cifar10_cpu_offloading](deepspeed/cifar10_cpu_offloading)    |           CIFAR-10           |  PyTorch (DeepSpeed)  |

## Natural Language Processing (NLP)

|                     Example                      |  Dataset   | Framework |
| :----------------------------------------------: | :--------: | :-------: |
| [albert_squad_pytorch](nlp/albert_squad_pytorch) |   SQuAD    |  PyTorch  |
|    [bert_glue_pytorch](nlp/bert_glue_pytorch)    |    GLUE    |  PyTorch  |
|  [word_language_model](nlp/word_language_model)  | WikiText-2 |  PyTorch  |

## HP Search Benchmarks

|                                    Example                                    |        Dataset        | Framework |
| :---------------------------------------------------------------------------: | :-------------------: | :-------: |
|      [darts_cifar10_pytorch](hp_search_benchmarks/darts_cifar10_pytorch)      |       CIFAR-10        |  PyTorch  |
| [darts_penntreebank_pytorch](hp_search_benchmarks/darts_penntreebank_pytorch) | Penn Treebank Dataset |  PyTorch  |

## Neural Architecture Search (NAS)

|             Example              | Dataset | Framework |
| :------------------------------: | :-----: | :-------: |
| [gaea_pytorch](nas/gaea_pytorch) |  DARTS  |  PyTorch  |

## Meta Learning

|                               Example                                | Dataset  | Framework |
| :------------------------------------------------------------------: | :------: | :-------: |
| [protonet_omniglot_pytorch](meta_learning/protonet_omniglot_pytorch) | Omniglot |  PyTorch  |

## Generative Adversarial Networks (GAN)

| Example                                      |     Dataset      |       Framework       |
| :------------------------------------------- | :--------------: | :-------------------: |
| [dc_gan_tf_keras](gan/dcgan_tf_keras)        |      MNIST       | TensorFlow (tf.keras) |
| [gan_mnist_pytorch](gan/gan_mnist_pytorch)   |      MNIST       |        PyTorch        |
| [deepspeed_dcgan](deepspeed/deepspeed_dcgan) | MNIST / CIFAR-10 |  PyTorch (DeepSpeed)  |
| [pix2pix_tf_keras](gan/pix2pix_tf_keras)     |     pix2pix      | TensorFlow (tf.keras) |
| [cyclegan](gan/cyclegan)                     |   monet2photo    |        PyTorch        |

## Custom Reducers

|                                 Example                                 | Dataset | Framework |
| :---------------------------------------------------------------------: | :-----: | :-------: |
| [custom_reducers_mnist_pytorch](features/custom_reducers_mnist_pytorch) |  MNIST  |  PyTorch  |

## HP Search Constraints

|                                Example                                | Dataset | Framework |
| :-------------------------------------------------------------------: | :-----: | :-------: |
| [hp_constraints_mnist_pytorch](features/hp_constraints_mnist_pytorch) |  MNIST  |  PyTorch  |

## Custom Search Method

|                            Example                            | Dataset | Framework |
| :-----------------------------------------------------------: | :-----: | :-------: |
| [asha_search_method](custom_search_method/asha_search_method) |  MNIST  |  PyTorch  |
