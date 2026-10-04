package com.convaiinnovations.laya;

import com.convaiinnovations.laya.json.PythonJson;
import com.convaiinnovations.laya.lang.LanguageDetection;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Decides which laya checkpoint a request should go to. A port of {@code laya.router}.
 *
 * <p>Three checkpoints, and the choice between them is not a tuning question. The English
 * checkpoint does not degrade gently off English, it collapses: on 20-option MASSIVE intent it
 * scores 0.100 on Hindi and 0.103 on Korean against 0.050 for random guessing, and it reports high
 * confidence while doing so (ECE 0.855 on Hindi). The multilingual checkpoint gains 21 points on
 * non-English XNLI and loses about 6 on English suites. So script is the primary signal, and a
 * wrong route is a wrong answer delivered confidently.
 *
 * <p>{@code typed-decisions} is never chosen automatically unless you ask for it, with
 * {@link Builder#autoTaskDetection} or an explicit task: it is fine-tuned on four specific
 * workflows and should not be a silent default.
 *
 * <p><b>Precedence</b>, highest first: an explicit model, an explicit task, a detected
 * typed-decisions workflow (opt-in), an explicit language, a caller's language hint, the built-in
 * script and language detection, and finally the configured default.
 *
 * <p>This class decides; it does not load. {@link #route} runs no model and touches no disk, so it
 * is safe to call on every request and to test without a checkpoint.
 */
public final class Router {

    /** The hub repository that bundles all three checkpoints. */
    public static final String BUNDLE_REPO = "convaiinnovations/laya";

    /** One of the three checkpoints laya ships. */
    public enum Checkpoint {
        /** ModernBERT-large, 512 tokens, English only. */
        ENGLISH("english"),
        /** mmBERT-base, 1024 tokens, 100+ languages. */
        MULTILINGUAL("multilingual"),
        /** ModernBERT-large fine-tuned on the four typed-decisions workflows. */
        TYPED_DECISIONS("typed-decisions");

        private final String wire;

        Checkpoint(String wire) {
            this.wire = wire;
        }

        /** The name the reference uses, which is what appears in a reason string. */
        public String wireName() {
            return wire;
        }

        @Override
        public String toString() {
            return wire;
        }
    }

    /** Where a checkpoint lives: a repository, and a subfolder within it when it has one. */
    public record ModelSpec(String repo, String subfolder) {

        /** The readable id: {@code repo}, or {@code repo/subfolder} when there is one. */
        public String repoString() {
            return subfolder == null || subfolder.isEmpty() ? repo : repo + "/" + subfolder;
        }
    }

    /**
     * The routing outcome: which checkpoint, why, and what was detected.
     *
     * @param model      the checkpoint to use
     * @param repo       its readable repository id
     * @param reason     why, in the reference's words; this reaches API responses
     * @param detection  what the detector saw, or null when an explicit argument decided
     * @param workflow   the typed-decisions workflow the question ids match, or null
     */
    public record RouteDecision(Checkpoint model, String repo, String reason,
                                LanguageDetection.Analysis detection, String workflow) {
    }

    /**
     * A caller's hint about whether the English checkpoint can read a state.
     *
     * <p>The reference accepts either a language code or a callable taking the state; this one
     * interface covers both -- {@link #of} for a fixed code, a lambda for anything that has to
     * look at the state, such as a language-identification model.
     *
     * <p>Returning null abstains, and abstaining is a real answer: it falls through to detection
     * rather than pinning a checkpoint on no evidence. So does a code that names no language,
     * which is what {@code LANG=C} in a minimal container gives.
     */
    @FunctionalInterface
    public interface LanguageHint {

        /** The language code for this state, or null to abstain. */
        String codeFor(Object state);

        /** A hint that always reports the same code. */
        static LanguageHint of(String code) {
            return state -> code;
        }
    }

    /** Per-call routing arguments. Every field may be null, which means "not specified". */
    public record RouteOptions(String model, String task, String lang, LanguageHint langGuess) {

        /** No per-call arguments: route on the questions and the state alone. */
        public static RouteOptions none() {
            return new RouteOptions(null, null, null, null);
        }

        /** Pin the checkpoint by name or alias. Beats everything else. */
        public RouteOptions model(String value) {
            return new RouteOptions(value, task, lang, langGuess);
        }

        /** Pin by task name. Beats everything but an explicit model. */
        public RouteOptions task(String value) {
            return new RouteOptions(model, value, lang, langGuess);
        }

        /** The language of the state, if the caller knows it. */
        public RouteOptions lang(String value) {
            return new RouteOptions(model, task, value, langGuess);
        }

        /** A hint consulted after an explicit language and before detection. */
        public RouteOptions langGuess(LanguageHint value) {
            return new RouteOptions(model, task, lang, value);
        }
    }

    // Names people are likely to type. Insertion order is not significant here, but the sorted
    // order is: an unknown name's error message lists these, and that message is recorded.
    private static final Map<String, Checkpoint> ALIASES;

    static {
        Map<String, Checkpoint> aliases = new LinkedHashMap<>();
        aliases.put("en", Checkpoint.ENGLISH);
        aliases.put("laya", Checkpoint.ENGLISH);
        aliases.put("default", Checkpoint.ENGLISH);
        aliases.put("multi", Checkpoint.MULTILINGUAL);
        aliases.put("ml", Checkpoint.MULTILINGUAL);
        aliases.put("laya-multilingual", Checkpoint.MULTILINGUAL);
        aliases.put("typed", Checkpoint.TYPED_DECISIONS);
        aliases.put("typed_decisions", Checkpoint.TYPED_DECISIONS);
        aliases.put("laya-typed-decisions", Checkpoint.TYPED_DECISIONS);
        aliases.put("decisions", Checkpoint.TYPED_DECISIONS);
        ALIASES = Collections.unmodifiableMap(aliases);
    }

    /** The bundle: one repository, the checkpoint in a subfolder. Only that subfolder downloads. */
    private static final Map<Checkpoint, ModelSpec> BUNDLED;

    /** The same checkpoints in their own repositories, for anyone who prefers them. */
    private static final Map<Checkpoint, ModelSpec> STANDALONE;

    static {
        Map<Checkpoint, ModelSpec> bundled = new LinkedHashMap<>();
        bundled.put(Checkpoint.ENGLISH, new ModelSpec(BUNDLE_REPO, null));
        bundled.put(Checkpoint.MULTILINGUAL, new ModelSpec(BUNDLE_REPO, "multilingual"));
        bundled.put(Checkpoint.TYPED_DECISIONS, new ModelSpec(BUNDLE_REPO, "typed-decisions"));
        BUNDLED = Collections.unmodifiableMap(bundled);

        Map<Checkpoint, ModelSpec> standalone = new LinkedHashMap<>();
        standalone.put(Checkpoint.ENGLISH, new ModelSpec("convaiinnovations/laya", null));
        standalone.put(Checkpoint.MULTILINGUAL,
                new ModelSpec("convaiinnovations/laya-multilingual", null));
        standalone.put(Checkpoint.TYPED_DECISIONS,
                new ModelSpec("convaiinnovations/laya-typed-decisions", null));
        STANDALONE = Collections.unmodifiableMap(standalone);
    }

    /**
     * The question-id signatures of the four typed-decisions workflows.
     *
     * <p>Matched exactly, never as a subset, so an unrelated schema that happens to contain
     * {@code urgency} is not captured.
     */
    private static final Map<String, Set<String>> TYPED_DECISION_WORKFLOWS;

    static {
        Map<String, Set<String>> workflows = new LinkedHashMap<>();
        workflows.put("agent_trace_observability",
                Set.of("action", "needs_review", "outcome", "risk", "urgency"));
        workflows.put("customer_service",
                Set.of("action", "category", "churn_risk", "needs_human", "urgency"));
        workflows.put("invoice_processing", Set.of("discrepancy_severity", "disposition",
                "duplicate", "matches_order", "urgency"));
        workflows.put("security_incidents", Set.of("credential_compromise", "disposition",
                "severity", "true_positive", "urgency"));
        TYPED_DECISION_WORKFLOWS = Collections.unmodifiableMap(workflows);
    }

    /** Subtags that mean "the English checkpoint can read this". */
    private static final Set<String> ENGLISH_SUBTAGS = Set.of("en", "eng", "english");

    /**
     * Codes that are valid {@code $LANG} values but name no language, so they answer nothing.
     *
     * <p>{@code C}, {@code POSIX} and {@code C.UTF-8} are what minimal images ship --
     * {@code C.UTF-8} is the default in the official Python image -- and the ISO 639-2 special
     * codes say the same thing in the standard's own vocabulary: {@code und} undetermined,
     * {@code zxx} no linguistic content, {@code mul} multiple languages. They abstain rather than
     * forcing the multilingual checkpoint on English text.
     */
    private static final Set<String> LANGUAGE_AGNOSTIC_CODES =
            Set.of("c", "posix", "und", "zxx", "mul");

    private final Map<Checkpoint, ModelSpec> models;
    private final Checkpoint defaultCheckpoint;
    private final boolean autoTaskDetection;
    private final LanguageHint langGuess;

    private Router(Builder builder) {
        Map<Checkpoint, ModelSpec> resolved =
                new LinkedHashMap<>(builder.standaloneRepos ? STANDALONE : BUNDLED);
        resolved.putAll(builder.overrides);
        this.models = Collections.unmodifiableMap(resolved);
        this.defaultCheckpoint = builder.defaultCheckpoint;
        this.autoTaskDetection = builder.autoTaskDetection;
        this.langGuess = builder.langGuess;
    }

    /** A router with the reference's defaults: the bundle, English as default, no auto-detection. */
    public static Router withDefaults() {
        return builder().build();
    }

    /** A router to configure. */
    public static Builder builder() {
        return new Builder();
    }

    /** Configures a {@link Router}. */
    public static final class Builder {

        private final Map<Checkpoint, ModelSpec> overrides = new LinkedHashMap<>();
        private Checkpoint defaultCheckpoint = Checkpoint.ENGLISH;
        private boolean autoTaskDetection;
        private boolean standaloneRepos;
        private LanguageHint langGuess;

        private Builder() {
        }

        /**
         * Where to send a state nothing identifies.
         *
         * <p>English by default, which is the reference's choice. A deployment whose traffic is
         * mostly not English should set this to {@link Checkpoint#MULTILINGUAL}: an unidentified
         * Latin-script state is no evidence of English, and this is the only knob that says so.
         */
        public Builder defaultCheckpoint(Checkpoint value) {
            this.defaultCheckpoint = requireNonNull(value, "defaultCheckpoint");
            return this;
        }

        /**
         * Allow question ids alone to select {@code typed-decisions}. Off by default.
         *
         * <p>Off because that checkpoint is fine-tuned on four specific synthetic workflows, and
         * a schema whose ids happen to match one of them should not silently change model.
         */
        public Builder autoTaskDetection(boolean value) {
            this.autoTaskDetection = value;
            return this;
        }

        /** Use the standalone repositories instead of the bundle. */
        public Builder standaloneRepos(boolean value) {
            this.standaloneRepos = value;
            return this;
        }

        /** A hint applied to every request, consulted after a per-call one. */
        public Builder langGuess(LanguageHint value) {
            this.langGuess = value;
            return this;
        }

        /** Point one checkpoint somewhere else -- a mirror, or a local export. */
        public Builder model(Checkpoint checkpoint, String repo, String subfolder) {
            requireNonNull(checkpoint, "checkpoint");
            this.overrides.put(checkpoint, new ModelSpec(requireNonNull(repo, "repo"), subfolder));
            return this;
        }

        /** Build it. */
        public Router build() {
            return new Router(this);
        }
    }

    /** Where this router expects each checkpoint to live. */
    public Map<Checkpoint, ModelSpec> models() {
        return models;
    }

    /** Where a state nothing identifies goes. */
    public Checkpoint defaultCheckpoint() {
        return defaultCheckpoint;
    }

    /** Whether question ids alone may select {@code typed-decisions}. */
    public boolean autoTaskDetection() {
        return autoTaskDetection;
    }

    // ------------------------------------------------------------------ the registry

    /**
     * The checkpoint a name or alias means.
     *
     * <p>Case and surrounding whitespace are ignored, because these names are typed by hand and
     * read out of configuration files.
     *
     * @throws IllegalArgumentException for anything that is not a checkpoint or an alias, with the
     *     reference's message listing both sets
     */
    public static Checkpoint normaliseName(String name) {
        Checkpoint resolved = lookupName(name);
        if (resolved == null) {
            throw new IllegalArgumentException(String.format(
                    "unknown model %s; choose one of %s (or an alias: %s)",
                    PythonJson.repr(name), pythonList(checkpointNames()),
                    pythonList(new TreeSet<>(ALIASES.keySet()))));
        }
        return resolved;
    }

    /**
     * The registry spec for a checkpoint name or alias, or null when it is not one.
     *
     * <p>The non-throwing sibling of {@link #normaliseName}, for a caller that also accepts things
     * the registry knows nothing about -- a hub repository id, a local directory, an ONNX export.
     * Those are not errors there; they simply are not registry names.
     */
    public static ModelSpec resolveModelSpec(String name) {
        Checkpoint resolved = lookupName(name);
        return resolved == null ? null : BUNDLED.get(resolved);
    }

    private static Checkpoint lookupName(String name) {
        if (name == null) {
            return null;
        }
        String key = name.trim().toLowerCase(Locale.ROOT);
        Checkpoint alias = ALIASES.get(key);
        if (alias != null) {
            return alias;
        }
        for (Checkpoint candidate : Checkpoint.values()) {
            if (candidate.wireName().equals(key)) {
                return candidate;
            }
        }
        return null;
    }

    private static List<String> checkpointNames() {
        List<String> names = new ArrayList<>();
        for (Checkpoint candidate : Checkpoint.values()) {
            names.add(candidate.wireName());
        }
        Collections.sort(names);
        return names;
    }

    /** Python's {@code repr} of a list of strings, which is what its error message interpolates. */
    private static String pythonList(Collection<String> values) {
        StringBuilder out = new StringBuilder("[");
        boolean first = true;
        for (String value : values) {
            if (!first) {
                out.append(", ");
            }
            out.append(PythonJson.repr(value));
            first = false;
        }
        return out.append(']').toString();
    }

    /**
     * The typed-decisions workflow whose question ids these are, or null.
     *
     * <p>An exact id-set match. A superset is not a match either: a support schema that adds one
     * field of its own is not the synthetic workflow the checkpoint was tuned on, and capturing it
     * would silently change model.
     */
    public static String matchTypedDecisionsWorkflow(Map<String, ?> questions) {
        Set<String> ids = questions == null ? Set.of() : questions.keySet();
        for (Map.Entry<String, Set<String>> entry : TYPED_DECISION_WORKFLOWS.entrySet()) {
            if (entry.getValue().size() == ids.size() && entry.getValue().containsAll(ids)) {
                return entry.getKey();
            }
        }
        return null;
    }

    /** The four workflow signatures, by name. */
    public static Map<String, Set<String>> typedDecisionWorkflows() {
        return TYPED_DECISION_WORKFLOWS;
    }

    /**
     * An order-sensitive signature for a question schema, so states asking the same thing can
     * share a forward pass.
     *
     * <p>Order-sensitive at every level, because option order is positional: a choice between
     * {@code {a, b}} and one between {@code {b, a}} are different questions whose answers mean
     * different things, and they must not be grouped. See {@link Question#spec()} for what a
     * typed question can and cannot carry into this.
     */
    public static String questionSchema(Map<String, Question> questions) {
        Map<String, Object> specs = new LinkedHashMap<>();
        if (questions != null) {
            for (Map.Entry<String, Question> entry : questions.entrySet()) {
                specs.put(entry.getKey(),
                        entry.getValue() == null ? null : entry.getValue().spec());
            }
        }
        return PythonJson.dumps(specs);
    }

    /**
     * True or false for a language code, or null when the code identifies nothing.
     *
     * <p>Accepts the forms a caller has to hand: {@code en}, {@code EN}, {@code en-US}, the POSIX
     * {@code en_US} that {@code $LANG} holds, and {@code en_US.UTF-8}.
     *
     * <p>Null means "no usable hint", which is what lets a language-identification model abstain
     * -- and it is also what a code naming no language returns, so {@code LANG=C} falls through to
     * detection instead of pinning every request to one checkpoint.
     *
     * <p>Routing needs one bit, not a language id: is this English Latin text, or something the
     * English checkpoint cannot read. So every code that names some other language answers false.
     */
    public static Boolean englishFromCode(Object value) {
        if (value == null) {
            return null;
        }
        String code = String.valueOf(value).trim().toLowerCase(Locale.ROOT);
        if (code.isEmpty()) {
            return null;
        }
        int dot = code.indexOf('.');
        if (dot >= 0) {
            code = code.substring(0, dot);                 // en_US.UTF-8 -> en_US
        }
        String primary = code.replace('_', '-');
        int dash = primary.indexOf('-');
        if (dash >= 0) {
            primary = primary.substring(0, dash);          // en_US -> en
        }
        if (primary.isEmpty() || LANGUAGE_AGNOSTIC_CODES.contains(primary)) {
            return null;
        }
        return ENGLISH_SUBTAGS.contains(primary);
    }

    // ------------------------------------------------------------------ routing

    /** Decide where this state goes. */
    public RouteDecision route(Object state) {
        return route(state, null, RouteOptions.none());
    }

    /** Decide where this state goes, with the questions available for workflow detection. */
    public RouteDecision route(Object state, Map<String, Question> questions) {
        return route(state, questions, RouteOptions.none());
    }

    /**
     * Decide where this state goes.
     *
     * <p>Runs no model and reads no disk. Precedence, highest first: an explicit model, an
     * explicit task, a detected workflow when {@link Builder#autoTaskDetection} is on, an explicit
     * language, a per-call hint, the router's installed hint, detection, then the default.
     */
    public RouteDecision route(Object state, Map<String, Question> questions,
            RouteOptions options) {
        RouteOptions settings = options == null ? RouteOptions.none() : options;

        if (settings.model() != null) {
            Checkpoint key = normaliseName(settings.model());
            return decision(key, "explicit model=" + PythonJson.repr(settings.model()), null,
                    null);
        }

        if (settings.task() != null) {
            String task = settings.task();
            // The reference accepts the task under either spelling, and reports back whichever
            // the caller wrote.
            Checkpoint key = "typed_decisions".equals(
                    task.toLowerCase(Locale.ROOT).replace('-', '_'))
                    ? Checkpoint.TYPED_DECISIONS
                    : normaliseName(task);
            return decision(key, "explicit task=" + PythonJson.repr(task), null, null);
        }

        // Computed here, and reported from here on even when a later branch decides: a caller
        // that pinned a language still wants to know its schema was a known workflow.
        String workflow = matchTypedDecisionsWorkflow(questions);
        if (workflow != null && autoTaskDetection) {
            return decision(Checkpoint.TYPED_DECISIONS,
                    "question ids match the " + PythonJson.repr(workflow)
                    + " typed-decisions workflow", null, workflow);
        }

        if (settings.lang() != null) {
            // Decisive only when the code names a language. Blank or whitespace is no usable
            // hint, so it falls through exactly as an abstaining hint does; a real code routes.
            Boolean resolved = englishFromCode(settings.lang());
            if (resolved != null) {
                Checkpoint key = resolved ? Checkpoint.ENGLISH : Checkpoint.MULTILINGUAL;
                return decision(key, "explicit lang=" + PythonJson.repr(settings.lang()), null,
                        workflow);
            }
        }

        // The caller's hint: the per-call one first, then the one installed on the router. Only a
        // hint that actually answers routes here; anything else falls through to detection.
        RouteDecision hinted = fromHint("lang_guess", settings.langGuess(), state, workflow);
        if (hinted != null) {
            return hinted;
        }
        hinted = fromHint("Router(lang_guess=...)", langGuess, state, workflow);
        if (hinted != null) {
            return hinted;
        }

        LanguageDetection.Analysis detection = LanguageDetection.analyse(state);
        Checkpoint key;
        String reason;
        if ("unknown".equals(detection.script())) {
            key = defaultCheckpoint;
            reason = "no letters detected in state; using default (" + key.wireName() + ")";
        } else if (!"latin".equals(detection.script())) {
            key = Checkpoint.MULTILINGUAL;
            reason = "non-Latin script (" + detection.script() + ", "
                    + PythonJson.percent0(detection.nonLatinFraction())
                    + "% of letters); the English checkpoint cannot read it";
        } else if (!detection.english()) {
            key = Checkpoint.MULTILINGUAL;
            if (detection.mixedSegment() != null && !detection.mixedSegment().isEmpty()) {
                reason = "Latin script, mostly English, but a line or field reads as "
                        + PythonJson.repr(detection.language()) + " ("
                        + PythonJson.repr(headCodePoints(detection.mixedSegment(), 60))
                        + "); the English checkpoint cannot read it";
            } else if (detection.language() != null) {
                reason = "Latin script but language looks like "
                        + PythonJson.repr(detection.language()) + ", not English";
            } else {
                // An unidentified Latin-script language, routed on the non-English letters alone,
                // because no stopword list here covers it.
                reason = "Latin script, language not identified but "
                        + PythonJson.percent0(detection.diacriticRate())
                        + "% non-English letters; not safe for the English checkpoint";
            }
        } else if (detection.languageUndecided()) {
            // Nothing identifies the language: too short, or only content words. That is no
            // evidence of English either, so it takes the same default as a state with no letters.
            key = defaultCheckpoint;
            reason = "Latin script, language not identified and no non-English letters; "
                    + "using default (" + key.wireName() + ")";
        } else {
            key = Checkpoint.ENGLISH;
            reason = "English Latin text";
        }
        return decision(key, reason, detection, workflow);
    }

    private RouteDecision fromHint(String source, LanguageHint hint, Object state,
            String workflow) {
        if (hint == null) {
            return null;
        }
        Boolean resolved = englishFromCode(hint.codeFor(state));
        if (resolved == null) {
            return null;
        }
        Checkpoint key = resolved ? Checkpoint.ENGLISH : Checkpoint.MULTILINGUAL;
        return decision(key, source + ": the caller identified this as "
                + (resolved ? "English" : "non-English") + " text", null, workflow);
    }

    private RouteDecision decision(Checkpoint key, String reason,
            LanguageDetection.Analysis detection, String workflow) {
        return new RouteDecision(key, models.get(key).repoString(), reason, detection, workflow);
    }

    /** The first {@code count} code points, which is what the reference's slice takes. */
    private static String headCodePoints(String text, int count) {
        if (text.codePointCount(0, text.length()) <= count) {
            return text;
        }
        return text.substring(0, text.offsetByCodePoints(0, count));
    }

    /** The checkpoints this router knows, in declaration order. */
    public static List<Checkpoint> checkpoints() {
        return List.of(Checkpoint.values());
    }

    /** The alias table, for a caller that wants to show the accepted names. */
    public static Set<String> aliases() {
        return new LinkedHashSet<>(ALIASES.keySet());
    }

    private static <T> T requireNonNull(T value, String what) {
        if (value == null) {
            throw new IllegalArgumentException(what + " must not be null");
        }
        return value;
    }
}
