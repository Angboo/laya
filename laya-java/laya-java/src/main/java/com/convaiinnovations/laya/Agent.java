package com.convaiinnovations.laya;

import ai.onnxruntime.OrtException;
import com.convaiinnovations.laya.config.AgentConfig;
import com.convaiinnovations.laya.decode.Decoder;
import com.convaiinnovations.laya.onnx.LayaSession;
import com.convaiinnovations.laya.sequence.Collator;
import com.convaiinnovations.laya.sequence.SequenceBuilder;
import com.convaiinnovations.laya.tokenizer.Tokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The runtime: ask several questions about one piece of evidence, in one forward pass.
 *
 * <p>laya is a "System 1" decision model -- a bidirectional encoder with a typed head, not a
 * generator. Every question about the same state becomes one row of a single batch, so four
 * questions cost one encode of the shared document rather than four.
 *
 * <pre>{@code
 * try (Agent agent = Agent.open(modelDir, graphDir)) {
 *     Map<String, Question> questions = new LinkedHashMap<>();
 *     questions.put("intent", Question.choice("What does the customer want?",
 *             new LinkedHashMap<>(Map.of("refund", "money back"))));
 *     Map<String, Answer> answers = agent.predict(email, questions, "en");
 * }
 * }</pre>
 *
 * <p>Pass questions in a {@link LinkedHashMap}: the answers come back keyed by question id, but a
 * choice's options are positional, so iteration order is part of what is asked.
 */
public final class Agent implements AutoCloseable {

    private final Tokenizer tokenizer;
    private final AgentConfig config;
    private final LayaSession session;
    private final int padId;

    private Agent(Tokenizer tokenizer, AgentConfig config, LayaSession session, int padId) {
        this.tokenizer = tokenizer;
        this.config = config;
        this.session = session;
        this.padId = padId;
    }

    /**
     * Opens a checkpoint.
     *
     * @param modelDirectory holds {@code rl_agent_config.json} and {@code tokenizer/}
     * @param graphDirectory holds {@code laya.onnx}, or {@code encoder.onnx} and {@code head.onnx}
     */
    public static Agent open(Path modelDirectory, Path graphDirectory)
            throws IOException, OrtException {
        return open(modelDirectory, graphDirectory, 1);
    }

    /** Opens a checkpoint with an explicit intra-op thread count. */
    public static Agent open(Path modelDirectory, Path graphDirectory, int threads)
            throws IOException, OrtException {
        Tokenizer tokenizer = Tokenizer.fromModelDirectory(modelDirectory);
        AgentConfig config = AgentConfig.fromModelDirectory(modelDirectory);
        LayaSession session = LayaSession.open(graphDirectory, threads);
        // Padding is masked out of attention but still embedded, so it has to be a real id.
        int padId = tokenizer.padId().orElseGet(
                () -> tokenizer.sepId().orElseThrow(() -> new IllegalStateException(
                        "this checkpoint names neither a pad_token nor a sep_token, so a batch "
                        + "of more than one question cannot be padded")));
        return new Agent(tokenizer, config, session, padId);
    }

    /** The checkpoint's tokenizer, for callers that want to measure a state's token cost. */
    public Tokenizer tokenizer() {
        return tokenizer;
    }

    /** The checkpoint's budgets and temperatures. */
    public AgentConfig config() {
        return config;
    }

    /** Asks every question about {@code state} in one forward pass, with no language override. */
    public Map<String, Answer> predict(Object state, Map<String, Question> questions)
            throws OrtException {
        return predict(state, questions, null);
    }

    /**
     * Asks every question about {@code state} in one forward pass.
     *
     * @param language a BCP-47-ish tag whose prefix may select a temperature override, or null
     * @return the answers, keyed by question id, in the order the questions were given
     */
    public Map<String, Answer> predict(Object state, Map<String, Question> questions,
                                       String language) throws OrtException {
        if (questions.isEmpty()) {
            return Map.of();
        }
        // The state is serialised and tokenized ONCE and reused for every question, which is what
        // `build_sequence`'s `state_ids` parameter exists for: the alternative re-tokenizes the
        // same document per question, and a document is usually far longer than a question.
        int[] stateIds = tokenizer.encode(
                SequenceBuilder.serializeState(state).replace(maskToken(), " "), -1);

        List<String> ids = new ArrayList<>(questions.keySet());
        List<Collator.Item> items = new ArrayList<>(ids.size());
        List<SequenceBuilder.Sequence> built = new ArrayList<>(ids.size());
        for (String id : ids) {
            Question question = questions.get(id);
            SequenceBuilder.Sequence sequence = SequenceBuilder.build(
                    tokenizer, state, question, config.maxLen(), config.headMaxLen(),
                    null, false, stateIds);
            built.add(sequence);
            items.add(new Collator.Item(sequence.ids(), sequence.markers(),
                    question.type().code()));
        }

        Collator.Batch batch = Collator.collate(items, padId);
        LayaSession.Output output = session.run(batch);

        Map<String, Answer> answers = new LinkedHashMap<>();
        for (int row = 0; row < ids.size(); row++) {
            String id = ids.get(row);
            // The FILTERED marker count: a head that overran `max_len` loses markers, and the
            // answer must be derived over the options that actually survived into the sequence.
            int optionCount = built.get(row).markers().length;
            float[] actionProbabilities =
                    Decoder.actionProbabilities(output.actLogits()[row]);
            answers.put(id, Decoder.decode(questions.get(id), output.logits()[row], optionCount,
                    actionProbabilities, null, language, config));
        }
        return Collections.unmodifiableMap(answers);
    }

    private String maskToken() {
        String mask = tokenizer.maskToken();
        if (mask == null) {
            throw new IllegalStateException(
                    "this checkpoint's tokenizer_config.json names no mask_token");
        }
        return mask;
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
