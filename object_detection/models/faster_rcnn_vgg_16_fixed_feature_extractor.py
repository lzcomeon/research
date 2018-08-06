import tensorflow as tf
import tensorflow.contrib.slim as slim
from object_detection.meta_architectures import faster_rcnn_meta_arch

Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dense = tf.keras.layers.Dense

class FasterRCNNVGG16FeatureExtractor(faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
    '''
    VGG-16 Faster RCNN Feature Extractor
    '''

    def __init__(self,
                 is_training,
                 first_stage_features_stride,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        super(FasterRCNNVGG16FeatureExtractor, self).__init__(
            is_training, first_stage_features_stride, batch_norm_trainable,
            reuse_weights, weight_decay
        )

    def preprocess(self, resized_inputs):
        '''
        Faster RCNN vgg-16 preprocessing
        :param resized_inputs: A [batch, height_in,width_in, channels] float32 tensor
        :return: preprocessed_inputs: A [batch, height_in,width_in, channels] float32 tensor
        '''

        channel_mean = [123.68, 115.779, 103.939]
        return resized_inputs - [[channel_mean]]

    def _extract_proposal_features(self, preprocessed_inputs, scope):
        '''Extracts first stage RPN features, in other word, feature map
        :param preprocessed_inputs: A [batch, height, width, channels] float32 tensor
            representing a batch of images.
        :param scope: A scope name. (unused)
        :return:
        rpn_feature_map: A tensor with shape [batch, height, width, depth]
        '''

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(preprocessed_inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        return (x)

    def _extract_box_classifier_features(self, proposal_feature_maps, scope):
        '''Extracts second stage box classifier features,(fulley connected layers)

        :param proposal_feature_maps:
        A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal
        :param scope:
        :return:
        proposal_classifier_features: A 4-D float tensor with shape
        [batch_size*self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal
        '''

        x = Dense(4096, activation='relu', name='fc1')(proposal_feature_maps)
        x = slim.dropout(x, 0.5, scope="Dropout_1", is_training=self._is_training)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        proposal_classifier_features = slim.dropout(x, 0.5, scope='Dropout_2', is_training=self._is_training)

        return(proposal_classifier_features)
        

