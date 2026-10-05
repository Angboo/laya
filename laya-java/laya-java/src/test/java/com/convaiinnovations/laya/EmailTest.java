package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import com.convaiinnovations.laya.lang.UnicodeTables;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * {@code laya.email}, against expectations recorded by running the reference.
 *
 * <p>The corpus is not a sample. Every case is a rule the module's comments argue for: a marker
 * that must fire, and -- more of them -- a near-miss that must NOT, because a cleaner that is too
 * eager deletes the sender's request, which is worse than leaving boilerplate behind. Eight of the
 * cases exist because a mutant survived without them, and they are named so that a later edit
 * which "simplifies" one of those rules has something to fail.
 */
final class EmailTest {

    @SuppressWarnings("unchecked")
    private static List<Object> rows(String key) {
        return (List<Object>) Fixtures.load("email.json").get(key);
    }

    @TestFactory
    @DisplayName("every cleaned body is byte-identical to the reference")
    List<DynamicTest> cleanedMatches() {
        List<Object> cases = rows("cleaned");
        assertTrue(cases.size() >= 70,
                () -> "the email corpus lost cases: " + cases.size() + " of at least 70");
        List<DynamicTest> tests = new ArrayList<>();
        for (Object entry : cases) {
            Map<String, Object> row = asMap(entry);
            String name = (String) row.get("name");
            tests.add(dynamicTest(name, () -> assertEquals(
                    row.get("result"), LayaEmail.cleanEmailBody((String) row.get("body")),
                    () -> "clean_email_body diverged on " + name)));
        }
        return tests;
    }

    @TestFactory
    @DisplayName("every budget cuts where the reference cuts")
    List<DynamicTest> budgetsMatch() {
        List<Object> cases = rows("budgets");
        assertTrue(cases.size() >= 5, "the budget cases vanished");
        List<DynamicTest> tests = new ArrayList<>();
        for (Object entry : cases) {
            Map<String, Object> row = asMap(entry);
            String name = (String) row.get("name");
            int budget = ((Number) row.get("max_chars")).intValue();
            tests.add(dynamicTest(name + " (max " + budget + ")", () -> assertEquals(
                    row.get("result"), LayaEmail.cleanEmailBody((String) row.get("body"), budget),
                    () -> "clean_email_body diverged on " + name)));
        }
        return tests;
    }

    @TestFactory
    @DisplayName("every state matches the reference, field for field and in order")
    @SuppressWarnings("unchecked")
    List<DynamicTest> statesMatch() {
        List<Object> cases = rows("states");
        assertTrue(cases.size() >= 7, "the state cases vanished");
        List<DynamicTest> tests = new ArrayList<>();
        for (Object entry : cases) {
            Map<String, Object> row = asMap(entry);
            String name = (String) row.get("name");
            tests.add(dynamicTest(name, () -> {
                Map<String, Object> want = (Map<String, Object>) row.get("result");
                Map<String, Object> got = LayaEmail.emailState(
                        (String) row.get("subject"), (String) row.get("body"),
                        (String) row.get("sender"), (Boolean) row.get("clean"),
                        ((Number) row.get("max_chars")).intValue(),
                        new LinkedHashMap<>(asMap(row.get("extra"))));
                assertEquals(want, got, () -> "email_state diverged on " + name);
                // Key order is not cosmetic: it is the order the model is shown the fields in.
                assertEquals(new ArrayList<>(want.keySet()), new ArrayList<>(got.keySet()),
                        () -> "email_state field order diverged on " + name);
            }));
        }
        return tests;
    }

    @Test
    @DisplayName("the question set is the same object graph as the preset")
    void questionsAreThePreset() {
        assertEquals(Presets.email().keySet(), LayaEmail.emailQuestions().keySet());
        assertEquals(Boolean.TRUE, Fixtures.load("email.json").get("questions_match_presets"),
                "the reference's re-export no longer equals laya.presets.email_questions");
    }

    // ------------------------------------------------------------------ the sign-off rule

    @Test
    @DisplayName("a name may begin with any Lu, Lt or Lo letter, and with nothing else")
    void signoffInitials() {
        // Each of these is a rule UPPER would get wrong; see UnicodeTablesTest for the counts.
        assertTrue(LayaEmail.isEnglishSignoff("Regards, Ana"));
        assertTrue(LayaEmail.isEnglishSignoff("Regards, Łukasz"), "Lu outside Latin-1");
        assertTrue(LayaEmail.isEnglishSignoff("Regards, 山田"), "Lo: a caseless script");
        assertTrue(LayaEmail.isEnglishSignoff("Regards, ǅarko"), "Lt: titlecase");
        assertFalse(LayaEmail.isEnglishSignoff("Thanks, Ⓐ"),
                "U+24B6 is Uppercase but category So, so it is not a letter");
        assertFalse(LayaEmail.isEnglishSignoff("Thanks, żaneta"), "a lowercase name is prose");
        assertFalse(LayaEmail.isEnglishSignoff("Thanks for the quick reply."));
        assertFalse(LayaEmail.isEnglishSignoff("Regards, Ana Maria Souza Lima"),
                "the tail allows at most three tokens");
    }

    @Test
    @DisplayName("a mark rides on its base, and a mark without one still separates tokens")
    void marksAreDroppedOnlyWithABase() {
        assertEquals("Jose", LayaEmail.dropMarks("José"), "a mark with a base goes");
        assertEquals("Jose", LayaEmail.dropMarks("Jo͏se"),
                "including U+034F, whose canonical combining class is zero");
        assertEquals("́Ana", LayaEmail.dropMarks("́Ana"),
                "a mark opening the string has no base");
        assertEquals("a ́b", LayaEmail.dropMarks("a ́b"),
                "nor does one following a space");
        // and the consequence, which is the reason the rule exists
        assertTrue(LayaEmail.isEnglishSignoff("Regards, José"));
        assertFalse(LayaEmail.isEnglishSignoff("Regards, ́Ana"),
                "a kept mark is not a token opener, so this is not a sign-off");
    }

    // --------------------------------------------------- the two performance-critical equalities

    @Test
    @DisplayName("javaWord is exactly the JDK's own regex word class, over every code point")
    void javaWordEqualsThePattern() throws Exception {
        // It replaced a matcher call per character, which cost more than the rest of the cleaner.
        // The replacement is only safe while it answers identically, so that is asserted rather
        // than argued: this side of boundaryView has to be the JDK's answer, not ours.
        Method javaWord = LayaEmail.class.getDeclaredMethod("javaWord", int.class);
        javaWord.setAccessible(true);
        Pattern word = Pattern.compile("\\w", Pattern.UNICODE_CHARACTER_CLASS);
        int examined = 0;
        for (int cp = 0; cp < 0x110000; cp++) {
            if (cp >= 0xD800 && cp <= 0xDFFF) {
                continue;
            }
            examined++;
            boolean spelled = (Boolean) javaWord.invoke(null, cp);
            boolean pattern = word.matcher(new String(Character.toChars(cp))).matches();
            if (spelled != pattern) {
                assertEquals(pattern, spelled, String.format("U+%04X", cp));
            }
        }
        assertEquals(0x110000 - 2048, examined, "every code point but the surrogates");
    }

    @Test
    @DisplayName("boundaryView rewrites exactly where the two word classes disagree")
    void boundaryViewSubstitutes() {
        assertSame("plain ascii", LayaEmail.boundaryView("plain ascii"),
                "an ASCII-only message is returned without copying");
        assertSame("café résumé", LayaEmail.boundaryView("café résumé"),
                "ordinary accented letters are word characters in both, so nothing is rewritten");
        // Java says word, Python says not -> U+0000, a non-word character to Java too.
        assertEquals("a b", LayaEmail.boundaryView("áb"), "a combining mark");
        assertEquals("a b", LayaEmail.boundaryView("a‌b"), "ZWNJ, a join control");
        assertEquals("a b", LayaEmail.boundaryView("a‍b"), "ZWJ");
        // Python says word, Java says not -> '0', which no literal in any pattern contains.
        assertEquals("a0b", LayaEmail.boundaryView("a½b"), "U+00BD is No: Python's word, not Java's");
        // Length is preserved, so the [^.]{0,N} windows count the same characters.
        for (String text : List.of("áb", "a‌b", "a½b", "x́½‌y")) {
            assertEquals(text.length(), LayaEmail.boundaryView(text).length(),
                    () -> "boundaryView changed the length of " + text);
        }
    }

    @Test
    @DisplayName("neither substitute can appear in a pattern literal")
    void substitutesAreNotLiterals() throws Exception {
        // The whole argument for substituting is that the replacement cannot complete or break a
        // literal. That holds only while no pattern here holds a digit or a NUL as a literal, so
        // it is checked rather than asserted in a comment.
        for (Pattern pattern : allPatterns()) {
            String source = pattern.pattern();
            assertFalse(source.indexOf(' ') >= 0,
                    () -> "a pattern holds a literal NUL: " + source.substring(0, 40));
        }
        // A digit may only appear inside the generated \d class or as a quantifier bound, never
        // as a literal to be matched, so the DISCLAIMER branches are checked directly.
        Pattern disclaimer = (Pattern) patternField("DISCLAIMER");
        assertFalse(disclaimer.pattern().matches(".*[^{,0-9]\\d[^}0-9,].*"),
                "a digit became a literal in DISCLAIMER; '0' is no longer a safe substitute");
    }

    @Test
    @DisplayName("the disclaimer filter never rejects what the pattern would accept")
    void hintIsANecessaryCondition() throws Exception {
        Pattern hint = (Pattern) patternField("DISCLAIMER_HINT");
        Pattern disclaimer = (Pattern) patternField("DISCLAIMER");
        // A filter is only sound as a NECESSARY condition: a false positive costs time, a false
        // negative keeps a disclaimer in the model's input. Checked over every recorded case,
        // which includes one example of every branch of the pattern.
        int fired = 0;
        for (Object entry : rows("cleaned")) {
            Map<String, Object> row = asMap(entry);
            String body = (String) row.get("body");
            for (String raw : body.split("\n", -1)) {
                final String paragraph = raw;
                if (disclaimer.matcher(LayaEmail.boundaryView(paragraph)).find()) {
                    fired++;
                    assertTrue(hint.matcher(paragraph).find(),
                            () -> "the filter rejected a paragraph the pattern accepts: "
                                  + paragraph);
                }
            }
        }
        final int hits = fired;
        assertTrue(hits >= 9,
                () -> "the corpus no longer exercises the disclaimer pattern: " + hits + " hits");

        // The corpus count is a floor, not coverage. Every BRANCH of the pattern is named here,
        // so a branch whose required literal is missing from the filter fails this rather than
        // waiting for a message in production to be the first to notice.
        String[] oneExamplePerBranch = {
            "This email and its contents are confidential and intended solely for the addressee.",
            "This message is confidential and may also privileged.",
            "If you have received this email in error please delete it.",
            "Esta mensagem e seus anexos sao confidenciais e de uso exclusivo do destinatario.",
            "Este mensaje es confidencial y para uso exclusivo del destinatario.",
            "Uso exclusivo do destinatario desta mensagem.",
            "Se recebeu esta mensagem por engano, apague-a.",
            "Usted ha recibido este mensaje por error.",
            "Antes de imprimir pense no meio ambiente.",
            "Pense no meio ambiente antes de imprimir.",
            "Ce message est confidentiel et destine uniquement au destinataire.",
            "Vous avez recu ce message par erreur.",
            "Usage exclusif du destinataire.",
        };
        for (String example : oneExamplePerBranch) {
            assertTrue(disclaimer.matcher(LayaEmail.boundaryView(example)).find(),
                    () -> "this no longer matches DISCLAIMER at all: " + example);
            assertTrue(hint.matcher(example).find(),
                    () -> "the filter would reject a disclaimer the pattern accepts: " + example);
        }
    }

    // ------------------------------------------------------------------------- the state

    @Test
    @DisplayName("a null or empty sender adds no field, and a null extra is dropped")
    void stateOmissions() {
        assertFalse(LayaEmail.emailState("s", "b", null).containsKey("from"));
        assertFalse(LayaEmail.emailState("s", "b", "").containsKey("from"),
                "Python's `if sender:` is false for an empty string");
        assertTrue(LayaEmail.emailState("s", "b", "a@b.c").containsKey("from"));
        Map<String, Object> extra = new LinkedHashMap<>();
        extra.put("kept", 1);
        extra.put("dropped", null);
        Map<String, Object> state = LayaEmail.emailState("s", "b", null, true, 3000, extra);
        assertTrue(state.containsKey("kept"));
        assertFalse(state.containsKey("dropped"), "a null extra is dropped, as the reference drops it");
        assertEquals(List.of("subject", "body", "kept"), new ArrayList<>(state.keySet()));
    }

    @Test
    @DisplayName("a null subject or body is empty, not an exception")
    void stateNulls() {
        Map<String, Object> state = LayaEmail.emailState(null, null);
        assertEquals("", state.get("subject"));
        assertEquals("", state.get("body"));
        assertEquals("", LayaEmail.cleanEmailBody(null));
    }

    @Test
    @DisplayName("a non-positive budget returns nothing rather than throwing")
    void nonPositiveBudget() {
        // Python slices with max_chars * 4 and then max_chars, and a slice to a non-positive
        // bound is empty. In Java a negative length is an exception, so the case is explicit.
        assertEquals("", LayaEmail.cleanEmailBody("a message", 0));
        assertEquals("", LayaEmail.cleanEmailBody("a message", -5));
        assertEquals("a", LayaEmail.cleanEmailBody("a message", 1));
    }

    @Test
    @DisplayName("the strip used on a subject is Python's, not Java's trim")
    void subjectStripIsPythons() {
        // U+00A0 and U+2007 are whitespace to str.strip() and not to String.trim(). The same gap
        // sat in the router's checkpoint names and routed a caller to the wrong model.
        assertEquals("Refund", LayaEmail.emailState(" Refund ", "b").get("subject"));
        assertTrue(UnicodeTables.isSpace(0x00a0) && UnicodeTables.isSpace(0x2007),
                "the premise of the assertion above");
    }

    // ------------------------------------------------------------------------- helpers

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object value) {
        return (Map<String, Object>) value;
    }

    private static Object patternField(String name) throws Exception {
        Class<?> patterns = Class.forName("com.convaiinnovations.laya.LayaEmail$Patterns");
        Field field = patterns.getDeclaredField(name);
        field.setAccessible(true);
        return field.get(null);
    }

    private static List<Pattern> allPatterns() throws Exception {
        Class<?> patterns = Class.forName("com.convaiinnovations.laya.LayaEmail$Patterns");
        List<Pattern> out = new ArrayList<>();
        for (Field field : patterns.getDeclaredFields()) {
            field.setAccessible(true);
            Object value;
            try {
                value = field.get(null);
            } catch (IllegalAccessException problem) {
                continue;
            }
            if (value instanceof Pattern) {
                out.add((Pattern) value);
            } else if (value instanceof List<?>) {
                for (Object element : (List<?>) value) {
                    if (element instanceof Pattern) {
                        out.add((Pattern) element);
                    }
                }
            }
        }
        assertTrue(out.size() >= 15, () -> "only " + out.size() + " patterns found by reflection");
        return out;
    }
}
