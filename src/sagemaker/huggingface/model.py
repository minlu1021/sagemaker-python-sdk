# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Placeholder docstring"""
from __future__ import absolute_import

import json
import logging
import os
import sys
import tarfile
import tempfile

import sagemaker
from sagemaker import image_uris
from sagemaker.deserializers import JSONDeserializer
from sagemaker.fw_utils import (
    model_code_key_prefix,
    validate_version_or_image_args,
)
from sagemaker.model import FrameworkModel, MODEL_SERVER_WORKERS_PARAM_NAME
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.utils import (
    _extract_model,
    _save_model,
)

logger = logging.getLogger("sagemaker")

NEO_SUPPORT_HF_FRAMEWORKS = set(
    ["pytorch"]
)

class HuggingFacePredictor(Predictor):
    """A Predictor for inference against Hugging Face Endpoints.

    This is able to serialize Python lists, dictionaries, and numpy arrays to
    multidimensional tensors for Hugging Face inference.
    """

    def __init__(
        self,
        endpoint_name,
        sagemaker_session=None,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    ):
        """Initialize an ``HuggingFacePredictor``.

        Args:
            endpoint_name (str): The name of the endpoint to perform inference
                on.
            sagemaker_session (sagemaker.session.Session): Session object that
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            serializer (sagemaker.serializers.BaseSerializer): Optional. Default
                serializes input data to .npy format. Handles lists and numpy
                arrays.
            deserializer (sagemaker.deserializers.BaseDeserializer): Optional.
                Default parses the response from .npy format to numpy array.
        """
        super(HuggingFacePredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
        )


def _validate_pt_tf_versions(pytorch_version, tensorflow_version, image_uri):
    """Placeholder docstring"""

    if image_uri is not None:
        return

    if tensorflow_version is not None and pytorch_version is not None:
        raise ValueError(
            "tensorflow_version and pytorch_version are both not None. "
            "Specify only tensorflow_version or pytorch_version."
        )
    if tensorflow_version is None and pytorch_version is None:
        raise ValueError(
            "tensorflow_version and pytorch_version are both None. "
            "Specify either tensorflow_version or pytorch_version."
        )


class HuggingFaceModel(FrameworkModel):
    """A Hugging Face SageMaker ``Model`` that can be deployed to a SageMaker ``Endpoint``."""

    _framework_name = "huggingface"

    def __init__(
        self,
        role,
        model_data=None,
        entry_point=None,
        transformers_version=None,
        tensorflow_version=None,
        pytorch_version=None,
        py_version=None,
        image_uri=None,
        predictor_cls=HuggingFacePredictor,
        model_server_workers=None,
        **kwargs,
    ):
        """Initialize a HuggingFaceModel.

        Args:
            model_data (str): The Amazon S3 location of a SageMaker model data
                ``.tar.gz`` file.
            role (str): An AWS IAM role specified with either the name or full ARN. The Amazon
                SageMaker training jobs and APIs that create Amazon SageMaker
                endpoints use this role to access training data and model
                artifacts. After the endpoint is created, the inference code
                might use the IAM role, if it needs to access an AWS resource.
            entry_point (str): The absolute or relative path to the Python source
                file that should be executed as the entry point to model
                hosting. If ``source_dir`` is specified, then ``entry_point``
                must point to a file located at the root of ``source_dir``.
                Defaults to None.
            transformers_version (str): Transformers version you want to use for
                executing your model training code. Defaults to None. Required
                unless ``image_uri`` is provided.
            tensorflow_version (str): TensorFlow version you want to use for
                executing your inference code. Defaults to ``None``. Required unless
                ``pytorch_version`` is provided. List of supported versions:
                https://github.com/aws/sagemaker-python-sdk#huggingface-sagemaker-estimators.
            pytorch_version (str): PyTorch version you want to use for
                executing your inference code. Defaults to ``None``. Required unless
                ``tensorflow_version`` is provided. List of supported versions:
                https://github.com/aws/sagemaker-python-sdk#huggingface-sagemaker-estimators.
            py_version (str): Python version you want to use for executing your
                model training code. Defaults to ``None``. Required unless
                ``image_uri`` is provided.
            image_uri (str): A Docker image URI. Defaults to None. If not specified, a
                default image for PyTorch will be used. If ``framework_version``
                or ``py_version`` are ``None``, then ``image_uri`` is required. If
                also ``None``, then a ``ValueError`` will be raised.
            predictor_cls (callable[str, sagemaker.session.Session]): A function
                to call to create a predictor with an endpoint name and
                SageMaker ``Session``. If specified, ``deploy()`` returns the
                result of invoking this function on the created endpoint name.
            model_server_workers (int): Optional. The number of worker processes
                used by the inference server. If None, server will use one
                worker per vCPU.
            **kwargs: Keyword arguments passed to the superclass
                :class:`~sagemaker.model.FrameworkModel` and, subsequently, its
                superclass :class:`~sagemaker.model.Model`.

        .. tip::

            You can find additional parameters for initializing this class at
            :class:`~sagemaker.model.FrameworkModel` and
            :class:`~sagemaker.model.Model`.
        """
        validate_version_or_image_args(transformers_version, py_version, image_uri)
        _validate_pt_tf_versions(
            pytorch_version=pytorch_version,
            tensorflow_version=tensorflow_version,
            image_uri=image_uri,
        )
        if py_version == "py2":
            raise ValueError("py2 is not supported with HuggingFace images")
        self.framework_version = transformers_version
        self.pytorch_version = pytorch_version
        self.tensorflow_version = tensorflow_version
        self.py_version = py_version

        super(HuggingFaceModel, self).__init__(
            model_data, image_uri, role, entry_point, predictor_cls=predictor_cls, **kwargs
        )

        self.model_server_workers = model_server_workers

    def register(
        self,
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        model_package_name=None,
        model_package_group_name=None,
        image_uri=None,
        model_metrics=None,
        metadata_properties=None,
        marketplace_cert=False,
        approval_status=None,
        description=None,
    ):
        """Creates a model package for creating SageMaker models or listing on Marketplace.

        Args:
            content_types (list): The supported MIME types for the input data.
            response_types (list): The supported MIME types for the output data.
            inference_instances (list): A list of the instance types that are used to
                generate inferences in real-time.
            transform_instances (list): A list of the instance types on which a transformation
                job can be run or on which an endpoint can be deployed.
            model_package_name (str): Model Package name, exclusive to `model_package_group_name`,
                using `model_package_name` makes the Model Package un-versioned.
                Defaults to ``None``.
            model_package_group_name (str): Model Package Group name, exclusive to
                `model_package_name`, using `model_package_group_name` makes the Model Package
                versioned. Defaults to ``None``.
            image_uri (str): Inference image URI for the container. Model class' self.image will
                be used if it is None. Defaults to ``None``.
            model_metrics (ModelMetrics): ModelMetrics object. Defaults to ``None``.
            metadata_properties (MetadataProperties): MetadataProperties object.
                Defaults to ``None``.
            marketplace_cert (bool): A boolean value indicating if the Model Package is certified
                for AWS Marketplace. Defaults to ``False``.
            approval_status (str): Model Approval Status, values can be "Approved", "Rejected",
                or "PendingManualApproval". Defaults to ``PendingManualApproval``.
            description (str): Model Package description. Defaults to ``None``.

        Returns:
            A `sagemaker.model.ModelPackage` instance.
        """
        instance_type = inference_instances[0]
        self._init_sagemaker_session_if_does_not_exist(instance_type)

        if image_uri:
            self.image_uri = image_uri
        if not self.image_uri:
            self.image_uri = self.serving_image_uri(
                region_name=self.sagemaker_session.boto_session.region_name,
                instance_type=instance_type,
            )
        return super(HuggingFaceModel, self).register(
            content_types,
            response_types,
            inference_instances,
            transform_instances,
            model_package_name,
            model_package_group_name,
            image_uri,
            model_metrics,
            metadata_properties,
            marketplace_cert,
            approval_status,
            description,
        )

    def compile(
        self,
        target_instance_family,
        input_shape,
        output_path,
        role,
        tags=None,
        job_name=None,
        compile_max_run=15 * 60,
        framework=None,
        framework_version=None,
        target_platform_os=None,
        target_platform_arch=None,
        target_platform_accelerator=None,
        compiler_options=None,
    ):
        """Compile this ``Model`` with SageMaker Neo."""
        # Check torch or not, only support PyTorch right now
        framework = framework or self._framework()
        if framework is None or framework not in NEO_SUPPORT_HF_FRAMEWORKS:
            raise ValueError(
                "You must specify framework, allowed values {}".format(NEO_SUPPORT_HF_FRAMEWORKS)
            )
        if framework.lower() == "pytorch" and not self._is_torch_available():
            raise RuntimeError(
                "PyTorch should be installed. "
                "To install PyTorch, read the instructions at https://pytorch.org/."
            )

        self.model_data = self.prepare_model_for_compilation(self.model_data, input_shape, output_path)

        return super(HuggingFaceModel, self).compile(
            target_instance_family=target_instance_family,
            input_shape=input_shape,
            output_path=output_path,
            role=role,
            tags=tags,
            job_name=job_name,
            compile_max_run=compile_max_run,
            framework=framework,
            framework_version=framework_version,
            target_platform_os=target_platform_os,
            target_platform_arch=target_platform_arch,
            target_platform_accelerator=target_platform_accelerator,
            compiler_options=compiler_options,
        )

    def _is_torch_available(self):
        import importlib.util
        if sys.version_info < (3, 8):
            import importlib_metadata
        else:
            import importlib.metadata as importlib_metadata

        _torch_available = importlib.util.find_spec("torch") is not None
        if _torch_available:
            try:
                _torch_version = importlib_metadata.version("torch")
                logger.info(f"PyTorch version {_torch_version} available.")
            except importlib_metadata.PackageNotFoundError:
                _torch_available = False
        return _torch_available

    def prepare_model_for_compilation(self, model_data, input_shape, output_path):
        """
        Prepare the PyTorch model file compatible with Neo

        Args:
            model_data (str): The Amazon S3 location of a SageMaker model data
                ``.tar.gz`` file.
            input_shape (dict): Specifies the name and shape of the expected
                inputs for your trained model in json dictionary form, for
                example: {'data': [1,3,1024,1024]}, or {'var1': [1,1,28,28],
                'var2': [1,1,28,28]}
            output_path (str): Specifies where to store the compiled model

        Returns:
            str: repacked model data ``.tar.gz`` file.
        """
        import torch
        from transformers import AutoModelForSequenceClassification

        if model_data is None:
            return model_data

        bucket = self.bucket or self.sagemaker_session.default_bucket()
        with tempfile.TemporaryDirectory() as tmpdir:
            model = None
            extracted_model = _extract_model(model_data, self.sagemaker_session, tmpdir)
            model_artifacts = os.listdir(extracted_model)
            for model_file in model_artifacts:
                if model_file.endswith('.pth') or model_file.endswith('.pt'):
                    if model is not None:
                        raise ValueError('Exactly one .pt or .pth file is allowed for PyTorch models.')
                    model = os.path.join(extracted_model, model_file)
                elif model_file.endswith('.bin'):
                    model = os.path.join(extracted_model, model_file)
            if model is None:
                raise ValueError("No pt or pth or bin file found for PyTorch model. ")

            # Neo only accpet .pt and .pth model file
            if model.endswith('.bin'):
                if isinstance(input_shape, str):
                    input_shape = json.loads(input_shape)
                if isinstance(input_shape, dict):
                    shapes = [input_shape[k] for k in sorted(input_shape)]
                    inputs = [torch.zeros(shape, dtype = torch.int64) for shape in shapes]
                else:
                    raise ValueError('Specifies the name and shape of the expected inputs in json dictionary form')

                try:
                    hf_model = AutoModelForSequenceClassification.from_pretrained(extracted_model, return_dict=False)
                    hf_model.eval()
                except Exception as e:
                    logger.error("Failed to load PyTorch model: {}".format(e))
                    raise

                saved_model_file = os.path.join(extracted_model, "model.pth")
                try:
                    model_trace = torch.jit.trace(hf_model, inputs, strict=False)
                    model_trace.save(saved_model_file)
                except Exception as e:
                    logger.error("Failed to save PyTorch model: {}".format(e))
                    raise

                tmp_model_path = os.path.join(extracted_model, "temp-model.tar.gz")
                with tarfile.open(tmp_model_path, mode="w:gz") as t:
                    t.add(saved_model_file, arcname="model.pth")
                repacked_model_data = "/".join([output_path, "prep_compile/model.tar.gz"])
                _save_model(repacked_model_data, tmp_model_path, self.sagemaker_session, self.model_kms_key)

                return repacked_model_data

        return model_data

    def prepare_container_def(self, instance_type=None, accelerator_type=None):
        """A container definition with framework configuration set in model environment variables.

        Args:
            instance_type (str): The EC2 instance type to deploy this Model to.
                For example, 'ml.p2.xlarge'.
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model.

        Returns:
            dict[str, str]: A container definition object usable with the
            CreateModel API.
        """
        deploy_image = self.image_uri
        if not deploy_image:
            if instance_type is None:
                raise ValueError(
                    "Must supply either an instance type (for choosing CPU vs GPU) or an image URI."
                )

            region_name = self.sagemaker_session.boto_session.region_name
            deploy_image = self.serving_image_uri(
                region_name, instance_type, accelerator_type=accelerator_type
            )

        deploy_key_prefix = model_code_key_prefix(self.key_prefix, self.name, deploy_image)
        self._upload_code(deploy_key_prefix, repack=True)
        deploy_env = dict(self.env)
        deploy_env.update(self._framework_env_vars())

        if self.model_server_workers:
            deploy_env[MODEL_SERVER_WORKERS_PARAM_NAME.upper()] = str(self.model_server_workers)
        return sagemaker.container_def(
            deploy_image, self.repacked_model_data or self.model_data, deploy_env
        )

    def serving_image_uri(self, region_name, instance_type, accelerator_type=None):
        """Create a URI for the serving image.

        Args:
            region_name (str): AWS region where the image is uploaded.
            instance_type (str): SageMaker instance type. Used to determine device type
                (cpu/gpu/family-specific optimized).
            accelerator_type (str): The Elastic Inference accelerator type to
                deploy to the instance for loading and making inferences to the
                model.

        Returns:
            str: The appropriate image URI based on the given parameters.

        """
        if self.tensorflow_version is not None:  # pylint: disable=no-member
            base_framework_version = (
                f"tensorflow{self.tensorflow_version}"  # pylint: disable=no-member
            )
        else:
            base_framework_version = f"pytorch{self.pytorch_version}"  # pylint: disable=no-member
        return image_uris.retrieve(
            self._framework_name,
            region_name,
            version=self.framework_version,
            py_version=self.py_version,
            instance_type=instance_type,
            accelerator_type=accelerator_type,
            image_scope="inference",
            base_framework_version=base_framework_version,
        )
