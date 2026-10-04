package com.convaiinnovations.laya;

import com.convaiinnovations.laya.json.PythonJson;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * One decision to put to the model: its type, its instruction, and the options it chooses between.
 *
 * <p>Three types, and they are not interchangeable -- the type name is the first token of the
 * sequence and an integer input to the head ({@code choice=0, score=1, noul=2}).
 *
 * <p><b>Option order is part of the question.</b> For a {@code choice}, options are rendered in the
 * criteria map's <i>insertion</i> order, so two orders are two different questions and the answer's
 * probabilities are positional. That is why the criteria are held in a {@link LinkedHashMap} and
 * why {@link #choice} copies the caller's map in iteration order: accepting a {@code HashMap} and
 * keeping the reference would let the JVM's hash order decide what the model was asked.
 */
public final class Question {

    /** The question type, whose ordinal is the integer the head receives. */
    public enum Type {
        /** Pick one labelled option. */
        CHOICE("choice"),
        /** Rate on an ordered scale; the answer is the expected value over levels. */
        SCORE("score"),
        /** Does the statement hold? The answer is the probability that it does. */
        NOUL("noul");

        private final String wire;

        Type(String wire) {
            this.wire = wire;
        }

        /** The name that appears in the sequence text and keys the temperature tables. */
        public String wireName() {
            return wire;
        }

        /** The integer the head receives: {@code choice=0, score=1, noul=2}. */
        public int code() {
            return ordinal();
        }
    }

    private static final String DEFAULT_FALSE = "no, the statement does not hold";
    private static final String DEFAULT_TRUE = "yes, the statement holds";

    private final Type type;
    private final String instructions;
    private final Map<String, Object> criteriaMap;      // choice, noul
    private final List<Object> criteriaList;            // score
    private final String falseLabel;
    private final String trueLabel;

    private Question(Type type, String instructions, Map<String, Object> criteriaMap,
                     List<Object> criteriaList, String falseLabel, String trueLabel) {
        this.type = type;
        this.instructions = instructions;
        this.criteriaMap = criteriaMap;
        this.criteriaList = criteriaList;
        this.falseLabel = falseLabel;
        this.trueLabel = trueLabel;
    }

    /**
     * A choice between labelled options, rendered in the map's iteration order.
     *
     * <p>A criterion of {@code null} or {@code ""} means "no description", and the option renders
     * as its label alone. Only those two: {@code 0} and {@code false} are legitimate criterion
     * values and render as {@code "label: 0"} and {@code "label: false"}.
     */
    public static Question choice(String instructions, Map<String, Object> criteria) {
        require(instructions != null, "a question needs instructions");
        require(criteria != null && !criteria.isEmpty(), "a choice question needs criteria");
        return new Question(Type.CHOICE, instructions, new LinkedHashMap<>(criteria), null,
                null, null);
    }

    /** A score over ordered levels, rendered as {@code "level <i>: <criterion>"}. */
    public static Question score(String instructions, List<Object> levels) {
        require(instructions != null, "a question needs instructions");
        require(levels != null && !levels.isEmpty(), "a score question needs levels");
        return new Question(Type.SCORE, instructions, null, new ArrayList<>(levels), null, null);
    }

    /** Does the statement hold? Options are always {@code [false, true]}, in that order. */
    public static Question noul(String instructions) {
        return noul(instructions, null, null, null);
    }

    /**
     * A noul with optional per-side criteria and optional labels.
     *
     * <p>{@code labels} is accepted <b>only</b> here -- a choice or score that supplies labels is
     * rejected in Python and is not expressible here. Both labels must be non-blank after trimming
     * and must differ.
     */
    public static Question noul(String instructions, Object falseCriterion, Object trueCriterion,
                                Map<String, String> labels) {
        require(instructions != null, "a question needs instructions");
        String falseLabel = "false";
        String trueLabel = "true";
        if (labels != null) {
            require(labels.size() == 2 && labels.containsKey("false") && labels.containsKey("true"),
                    "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
            falseLabel = labels.get("false");
            trueLabel = labels.get("true");
            require(falseLabel != null && trueLabel != null,
                    "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
            falseLabel = falseLabel.trim();
            trueLabel = trueLabel.trim();
            require(!falseLabel.isEmpty() && !trueLabel.isEmpty() && !falseLabel.equals(trueLabel),
                    "noul labels must map exactly 'false' and 'true' to distinct non-empty strings");
        }
        Map<String, Object> criteria = new LinkedHashMap<>();
        criteria.put("false", falseCriterion);
        criteria.put("true", trueCriterion);
        return new Question(Type.NOUL, instructions, criteria, null, falseLabel, trueLabel);
    }

    public Type type() {
        return type;
    }

    public String instructions() {
        return instructions;
    }

    /**
     * The option texts, in label-index order.
     *
     * <p>This is the text the model actually reads, so it is the contract a port has to match
     * character for character: everything downstream is token ids derived from it.
     */
    public List<String> renderOptions() {
        List<String> out = new ArrayList<>();
        switch (type) {
            case CHOICE:
                for (Map.Entry<String, Object> entry : criteriaMap.entrySet()) {
                    Object value = entry.getValue();
                    // `null` and `""` only. A label with no description renders as itself.
                    out.add(value == null || "".equals(value)
                            ? entry.getKey()
                            : entry.getKey() + ": " + renderCriterion(value));
                }
                return out;
            case SCORE:
                for (int i = 0; i < criteriaList.size(); i++) {
                    out.add("level " + i + ": " + renderCriterion(criteriaList.get(i)));
                }
                return out;
            default:
                Object no = criteriaMap.get("false");
                Object yes = criteriaMap.get("true");
                out.add(falseLabel + ": "
                        + (no == null || "".equals(no) ? DEFAULT_FALSE : renderCriterion(no)));
                out.add(trueLabel + ": "
                        + (yes == null || "".equals(yes) ? DEFAULT_TRUE : renderCriterion(yes)));
                return out;
        }
    }

    /** The labels an answer is keyed by: the criteria keys, or the level indices, or false/true. */
    public List<String> labels() {
        List<String> out = new ArrayList<>();
        switch (type) {
            case CHOICE:
                out.addAll(criteriaMap.keySet());
                return out;
            case SCORE:
                for (int i = 0; i < criteriaList.size(); i++) {
                    out.add(Integer.toString(i));
                }
                return out;
            default:
                out.add(falseLabel);
                out.add(trueLabel);
                return out;
        }
    }

    /**
     * One criterion as text: a string passes through, anything else becomes compact JSON.
     *
     * <p>Python spells that JSON with {@code json.dumps(..., ensure_ascii=False,
     * separators=(", ", ": "))}, which is what {@link PythonJson} reproduces -- including float
     * formatting, because this text is tokenized and {@code 1e+16} and {@code 1.0E16} are different
     * token sequences.
     *
     * <p>Python passes {@code default=str}, so an object it cannot serialise becomes its
     * {@code str()}. Here an unsupported type raises instead: inventing text from a Java object's
     * {@code toString()} would put a class name and an identity hash into the model's input.
     */
    public static String renderCriterion(Object value) {
        return value instanceof String ? (String) value : PythonJson.dumps(value);
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }
}
