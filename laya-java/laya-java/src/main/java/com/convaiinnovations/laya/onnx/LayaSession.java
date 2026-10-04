package com.convaiinnovations.laya.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.convaiinnovations.laya.sequence.Collator;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The ONNX graph, in either of the two forms laya exports.
 *
 * <ul>
 *   <li><b>Fused</b> {@code laya.onnx}: one graph taking {@code input_ids}, {@code attention_mask},
 *       {@code marker_pos}, {@code marker_mask}, {@code qtype} and returning {@code logits} and
 *       {@code act_logits}. This is what {@code ONNXAgent} loads by default.
 *   <li><b>Split</b> {@code encoder.onnx} + {@code head.onnx}: the encoder returns
 *       {@code last_hidden_state}, which the head consumes with the marker inputs.
 * </ul>
 *
 * <p>Both are supported because both exist in the wild -- {@code laya-dotnet} carries the same pair
 * of paths -- and which one a directory holds is discoverable, so there is no reason to make the
 * caller say. Instances are not thread-safe for concurrent {@code run} calls on one session unless
 * ONNX Runtime is configured for it; hold one per worker or serialise access.
 */
public final class LayaSession implements AutoCloseable {

    /** One batch's head outputs. */
    public record Output(float[][] logits, float[][] actLogits) {
    }

    private final OrtEnvironment environment;
    private final OrtSession fused;
    private final OrtSession encoder;
    private final OrtSession head;

    private LayaSession(OrtEnvironment environment, OrtSession fused, OrtSession encoder,
                        OrtSession head) {
        this.environment = environment;
        this.fused = fused;
        this.encoder = encoder;
        this.head = head;
    }

    /**
     * Opens whichever graph form {@code directory} holds, preferring the fused one.
     *
     * @param threads intra-op threads; 1 is the right default for a request-per-thread server,
     *                where the parallelism is already in the requests
     */
    public static LayaSession open(Path directory, int threads) throws OrtException, IOException {
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(threads);
        Path fusedPath = directory.resolve("laya.onnx");
        if (Files.isRegularFile(fusedPath)) {
            return new LayaSession(environment,
                    environment.createSession(fusedPath.toString(), options), null, null);
        }
        Path encoderPath = directory.resolve("encoder.onnx");
        Path headPath = directory.resolve("head.onnx");
        if (Files.isRegularFile(encoderPath) && Files.isRegularFile(headPath)) {
            return new LayaSession(environment, null,
                    environment.createSession(encoderPath.toString(), options),
                    environment.createSession(headPath.toString(), options));
        }
        throw new IOException(
                "no laya graph in " + directory + ": expected laya.onnx, or encoder.onnx and "
                + "head.onnx");
    }

    /** Whether this session is the fused single-graph form. */
    public boolean isFused() {
        return fused != null;
    }

    /** Runs one collated batch. */
    public Output run(Collator.Batch batch) throws OrtException {
        List<OnnxTensor> owned = new ArrayList<>();
        try {
            OnnxTensor inputIds = track(owned, OnnxTensor.createTensor(environment, batch.inputIds()));
            OnnxTensor attention = track(owned, OnnxTensor.createTensor(environment, batch.attentionMask()));
            OnnxTensor markerPos = track(owned, OnnxTensor.createTensor(environment, batch.markerPos()));
            OnnxTensor markerMask = track(owned, OnnxTensor.createTensor(environment, batch.markerMask()));
            OnnxTensor qtype = track(owned, OnnxTensor.createTensor(environment, batch.qtype()));
            if (fused != null) {
                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("input_ids", inputIds);
                inputs.put("attention_mask", attention);
                inputs.put("marker_pos", markerPos);
                inputs.put("marker_mask", markerMask);
                inputs.put("qtype", qtype);
                try (OrtSession.Result result = fused.run(inputs, java.util.Set.of("logits", "act_logits"))) {
                    return new Output(floats(result, 0), floats(result, 1));
                }
            }
            Map<String, OnnxTensor> encoderInputs = new HashMap<>();
            encoderInputs.put("input_ids", inputIds);
            encoderInputs.put("attention_mask", attention);
            try (OrtSession.Result encoded = encoder.run(encoderInputs,
                    java.util.Set.of("last_hidden_state"))) {
                OnnxTensor hidden = (OnnxTensor) encoded.get(0);
                Map<String, OnnxTensor> headInputs = new HashMap<>();
                headInputs.put("hidden_states", hidden);
                headInputs.put("marker_pos", markerPos);
                headInputs.put("marker_mask", markerMask);
                headInputs.put("qtype", qtype);
                headInputs.put("attention_mask", attention);
                try (OrtSession.Result result = head.run(headInputs,
                        java.util.Set.of("logits", "act_logits"))) {
                    return new Output(floats(result, 0), floats(result, 1));
                }
            }
        } finally {
            for (OnnxTensor tensor : owned) {
                tensor.close();
            }
        }
    }

    private static OnnxTensor track(List<OnnxTensor> owned, OnnxTensor tensor) {
        owned.add(tensor);
        return tensor;
    }

    private static float[][] floats(OrtSession.Result result, int index) throws OrtException {
        return (float[][]) ((OnnxTensor) result.get(index)).getValue();
    }

    @Override
    public void close() throws OrtException {
        if (fused != null) {
            fused.close();
        }
        if (encoder != null) {
            encoder.close();
        }
        if (head != null) {
            head.close();
        }
    }
}
