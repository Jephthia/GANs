from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mimetypes
import os
import numpy as np
import tensorflow as tf
import h5py as h5

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
    plugin_name = "tensors"

    def __init__(self, context):
        self._multiplexer = context.multiplexer

    def get_plugin_apps(self):
        return {
            "/scalars": self.scalars_route,
            "/tensors": self.tensors_route,
            "/tags": self._serve_tags,
            "/static/*": self._serve_static_file,
        }

    @wrappers.Request.application
    def _serve_tags(self, request):
        plugin_runs = self._multiplexer.PluginRunToTagToContent(self.plugin_name)
        runs = {}

        for run, tags in plugin_runs.items():
            runs[run] = list(tags.keys())

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
        return bool(self._multiplexer.PluginRunToTagToContent(self.plugin_name))

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path="/static/index.js")

    @wrappers.Request.application
    def scalars_route(self, request):
        tag = request.args.get("tag")
        run = request.args.get("run")

        try:
            events = self._multiplexer.Tensors('.', 'test')
            result = { 'tag': tag, 'steps': {} }
            print('events', len(events))
            for event in events:
                print('step', event.step)
                tensor = tensor_util.make_ndarray(event.tensor_proto).tolist()
                result['steps'][event.step] = tensor
        except KeyError:
            raise errors.NotFoundError(f'No scalar data for run={run}, tag={tag}')

        return http_util.Respond(request, result, "application/json")

    @wrappers.Request.application
    def tensors_route(self, request):
        log_dir = request.args.get("log_dir")
        cursor = int(request.args.get("cursor"))
        limit = int(request.args.get("limit"))
        no_kernel = request.args.get("noKernel") == 'true'
        no_bias = request.args.get("noBias") == 'true'

        results = []

        with h5.File(log_dir, 'r') as f:
            # Go through the groups
            for group_name, datasets in f.items():
                result = { 'name': '', 'kernel': { 'steps': {} }, 'bias': { 'steps': {} } }
                result['name'] = group_name
                
                # Go through the steps of this group
                for step in range(cursor, cursor+limit):
                    if not no_kernel and str(step) in datasets['kernel']:
                        result['kernel']['steps'][step] = datasets['kernel'][str(step)][()].tolist()
                    if not no_bias and str(step) in datasets['bias']:
                        result['bias']['steps'][step] = datasets['bias'][str(step)][()].tolist()
                        
                results.append(result)

        return http_util.Respond(request, results, "application/json")