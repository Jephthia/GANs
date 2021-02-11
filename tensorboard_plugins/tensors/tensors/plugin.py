# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A sample plugin to demonstrate reading scalars."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mimetypes
import os
import numpy as np
import tensorflow as tf

import six
from werkzeug import wrappers

from tensorboard import errors
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.util import tensor_util
from tensorboard.plugins.scalar import metadata

_SCALAR_PLUGIN_NAME = metadata.PLUGIN_NAME
_PLUGIN_DIRECTORY_PATH_PART = "/data/plugin/tensors/"


class TensorsPlugin(base_plugin.TBPlugin):
    """Raw summary example plugin for TensorBoard."""

    plugin_name = "tensors"

    def __init__(self, context):
        """Instantiates ExampleRawScalarsPlugin.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._multiplexer = context.multiplexer

    def get_plugin_apps(self):
        return {
            "/scalars": self.scalars_route,
            "/tags": self._serve_tags,
            "/static/*": self._serve_static_file,
        }

    @wrappers.Request.application
    def _serve_tags(self, request):
        runs = {}

        for key, content in self._multiplexer.Runs().items():
            runs[key] = content['tensors']

        return http_util.Respond(request, runs, "application/json")

    @wrappers.Request.application
    def _serve_static_file(self, request):
        """Returns a resource file from the static asset directory.

        Requests from the frontend have a path in this form:
        /data/plugin/example_raw_scalars/static/foo
        This serves the appropriate asset: ./static/foo.

        Checks the normpath to guard against path traversal attacks.
        """
        static_path_part = request.path[len(_PLUGIN_DIRECTORY_PATH_PART) :]
        resource_name = os.path.normpath(
            os.path.join(*static_path_part.split("/"))
        )
        if not resource_name.startswith("static" + os.path.sep):
            return http_util.Respond(
                request, "Not found", "text/plain", code=404
            )

        resource_path = os.path.join(os.path.dirname(__file__), resource_name)
        with open(resource_path, "rb") as read_file:
            mimetype = mimetypes.guess_type(resource_path)[0]
            return http_util.Respond(
                request, read_file.read(), content_type=mimetype
            )

    def is_active(self):
        return bool(
            self._multiplexer.PluginRunToTagToContent(_SCALAR_PLUGIN_NAME)
        )

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path="/static/index.js")

    @wrappers.Request.application
    def scalars_route(self, request):
        tag = request.args.get("tag")
        run = request.args.get("run")

        try:
            events = self._multiplexer.Tensors(run, tag)
            result = { 'tag': tag, 'steps': {} }

            for event in events:
                tensor = tensor_util.make_ndarray(event.tensor_proto).tolist()
                result['steps'][event.step] = tensor
        except KeyError:
            raise errors.NotFoundError(f'No scalar data for run={run}, tag={tag}')

        return http_util.Respond(request, result, "application/json")