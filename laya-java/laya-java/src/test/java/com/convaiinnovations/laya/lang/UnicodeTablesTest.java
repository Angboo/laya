package com.convaiinnovations.laya.lang;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Fixtures;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;
import java.util.Map;
import java.util.function.IntPredicate;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The character tables, against CPython, over every code point there is.
 *
 * <p>A digest per property rather than a sampled corpus. These tables decide which script a
 * request is counted under, so "agrees on the cases we thought of" is the wrong assurance: each
 * test below compares a sha256 over all 1,114,112 code points against one recorded from CPython,
 * and fails the moment a table edit, a CPython bump or a JDK upgrade moves a single one of them.
 *
 * <p>Characters are built with {@link #ch} rather than written as unicode escapes. A
 * backslash-u escape in Java source is translated before the file is even lexed, so an escape for
 * a control or format character is a hazard in a source file rather than a literal in a string.
 */
class UnicodeTablesTest {

    private static final int MAX = 0x110000;

    /** One code point as a string, without putting a unicode escape in this source file. */
    private static String ch(int codePoint) {
        return new String(Character.toChars(codePoint));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> digests() {
        return (Map<String, Object>) Fixtures.load("lang_detect.json").get("unicode_digests");
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is required of every JVM", impossible);
        }
    }

    private static String hex(MessageDigest digest) {
        StringBuilder out = new StringBuilder(64);
        for (byte value : digest.digest()) {
            out.append(Character.forDigit((value >> 4) & 0xF, 16));
            out.append(Character.forDigit(value & 0xF, 16));
        }
        return out.toString();
    }

    /** The digest Python records for a boolean property: one byte per code point, in order. */
    private static String predicateDigest(IntPredicate predicate) {
        MessageDigest digest = sha256();
        byte[] one = {'1'};
        byte[] zero = {'0'};
        for (int cp = 0; cp < MAX; cp++) {
            digest.update(predicate.test(cp) ? one : zero);
        }
        return hex(digest);
    }

    private void assertPredicate(String name, IntPredicate predicate) {
        assertEquals(digests().get(name), predicateDigest(predicate),
                name + ": the compiled table no longer agrees with CPython over all of Unicode."
                + " Regenerate with scripts/gen_unicode_tables.py and read the diff before"
                + " committing it -- a moved code point changes a routing decision.");
    }

    @Test
    @DisplayName("the recorded Unicode version is the one the tables were built from")
    void version() {
        assertEquals(Fixtures.load("lang_detect.json").get("unicode_version"),
                UnicodeTables.UNICODE_VERSION);
    }

    @Test
    @DisplayName("isAlpha is Python's str.isalpha on every code point")
    void alpha() {
        assertPredicate("alpha", UnicodeTables::isAlpha);
    }

    @Test
    @DisplayName("isWordChar is Python's word-not-digit class on every code point")
    void word() {
        assertPredicate("word", UnicodeTables::isWordChar);
    }

    @Test
    @DisplayName("isCombining is a non-zero canonical combining class on every code point")
    void combining() {
        assertPredicate("combining", UnicodeTables::isCombining);
    }

    @Test
    @DisplayName("isDigit is Python's digit class on every code point")
    void digit() {
        assertPredicate("digit", UnicodeTables::isDigit);
    }

    @Test
    @DisplayName("isUpper is Python's str.isupper on every code point")
    void upper() {
        assertPredicate("upper", UnicodeTables::isUpper);
    }

    @Test
    @DisplayName("isLower is Python's str.islower on every code point")
    void lower() {
        assertPredicate("lower", UnicodeTables::isLower);
    }

    @Test
    @DisplayName("isSpace is Python's str.isspace on every code point")
    void space() {
        assertPredicate("space", UnicodeTables::isSpace);
    }

    @Test
    @DisplayName("isMark is the Mn/Mc/Me categories on every code point")
    void mark() {
        assertPredicate("mark", UnicodeTables::isMark);
    }

    @Test
    @DisplayName("isInitial is the Lu/Lt/Lo categories on every code point")
    void initial() {
        assertPredicate("initial", UnicodeTables::isInitial);
    }

    @Test
    @DisplayName("isMark is not the combining class, and isInitial is not str.isupper")
    void theTwoNewTablesAreNotTheOldOnes() {
        // Both were nearly implemented by reusing a table that already existed, and both reuses
        // would have been wrong on a large, specific set. Asserted so that a later simplification
        // has to argue with a number rather than with a comment.
        int markWithZeroCombiningClass = 0;
        int initialThatIsNotUpper = 0;
        int upperThatIsNotInitial = 0;
        for (int cp = 0; cp < 0x110000; cp++) {
            if (UnicodeTables.isMark(cp) && !UnicodeTables.isCombining(cp)) {
                markWithZeroCombiningClass++;
            }
            if (UnicodeTables.isInitial(cp) && !UnicodeTables.isUpper(cp)) {
                initialThatIsNotUpper++;
            }
            if (UnicodeTables.isUpper(cp) && !UnicodeTables.isInitial(cp)) {
                upperThatIsNotInitial++;
            }
        }
        assertEquals(1528, markWithZeroCombiningClass,
                "isCombining is the canonical combining CLASS; these are category M with class 0");
        assertEquals(131643, initialThatIsNotUpper,
                "every caseless script: str.isupper() is false for all of them");
        assertEquals(120, upperThatIsNotInitial,
                "the Uppercase property holds symbols that are not letters, such as U+24B6");
        // The three that make the difference observable in a sign-off's name.
        assertTrue(UnicodeTables.isInitial(0x01C5), "U+01C5 is Lt, a name may begin with it");
        assertFalse(UnicodeTables.isUpper(0x01C5), "and str.isupper() rejects it");
        assertTrue(UnicodeTables.isInitial(0x5C71), "U+5C71 is Lo, a name may begin with it");
        assertTrue(UnicodeTables.isUpper(0x24B6), "U+24B6 is Uppercase...");
        assertFalse(UnicodeTables.isInitial(0x24B6), "...but it is category So, not a letter");
        assertTrue(UnicodeTables.isMark(0x034F), "U+034F is Mn...");
        assertFalse(UnicodeTables.isCombining(0x034F), "...with a canonical combining class of 0");
    }

    @Test
    @DisplayName("pythonLower maps every code point the way Python's str.lower does")
    void pythonLower() {
        MessageDigest digest = sha256();
        StringBuilder row = new StringBuilder(32);
        for (int cp = 0; cp < MAX; cp++) {
            if (cp >= 0xD800 && cp <= 0xDFFF) {
                continue;             // a lone surrogate is not a character a String can hold
            }
            String mapped = UnicodeTables.pythonLower(ch(cp));
            row.setLength(0);
            row.append(cp).append(':');
            int i = 0;
            boolean first = true;
            while (i < mapped.length()) {
                int lowered = mapped.codePointAt(i);
                i += Character.charCount(lowered);
                if (!first) {
                    row.append(',');
                }
                row.append(lowered);
                first = false;
            }
            row.append(';');
            digest.update(row.toString().getBytes(StandardCharsets.US_ASCII));
        }
        assertEquals(digests().get("python_lower"), hex(digest),
                "pythonLower no longer agrees with CPython's str.lower. String.toLowerCase is not"
                + " a substitute: it applies the contextual final-sigma rule.");
    }

    @Test
    @DisplayName("the JDK's own predicates would NOT pass, which is why these tables exist")
    void theJdkDisagrees() {
        // Not a tautology: this is the measurement behind the comment on UnicodeTables, kept as a
        // test so the claim is checked rather than asserted. If a future JDK ever catches up, this
        // fails and the table can be reconsidered deliberately rather than kept out of habit.
        assertFalse(predicateDigest(Character::isLetter).equals(digests().get("alpha")),
                "Character.isLetter now agrees with CPython; the ALPHA table may be removable");
        assertFalse(predicateDigest(Character::isWhitespace).equals(digests().get("space")),
                "Character.isWhitespace now agrees with CPython's isspace");
        assertFalse(predicateDigest(Character::isUpperCase).equals(digests().get("upper")),
                "Character.isUpperCase now agrees with CPython's isupper");
        assertFalse(predicateDigest(cp -> {
            int type = Character.getType(cp);
            return type == Character.NON_SPACING_MARK || type == Character.COMBINING_SPACING_MARK;
        }).equals(digests().get("combining")),
                "the Mn/Mc approximation -- which the .NET port uses -- now agrees with the"
                + " canonical combining class");
    }

    @Test
    @DisplayName("strip and splitOnWhitespace follow isSpace, not Character.isWhitespace")
    void stripAndSplit() {
        // U+00A0 is whitespace to Python and not to the JDK, which is the whole point: a ticket
        // pasted out of a word processor is full of them, and splitting on the JDK's set fuses
        // two words into one token that no stopword list holds.
        String nbsp = ch(0x00A0);
        String nel = ch(0x0085);
        String narrowNbsp = ch(0x202F);
        String figureSpace = ch(0x2007);
        assertEquals("a" + nbsp + "b",
                UnicodeTables.strip(nel + " a" + nbsp + "b " + narrowNbsp));
        assertEquals(List.of("a", "b"),
                UnicodeTables.splitOnWhitespace("  a" + nbsp + nbsp + "b  "));
        assertEquals(List.of(), UnicodeTables.splitOnWhitespace(" " + figureSpace + nel + " "));
        assertTrue(UnicodeTables.isBlank(nbsp + figureSpace + narrowNbsp + nel));
        assertFalse(UnicodeTables.isBlank(ch(0x200B)));   // zero-width space is NOT isspace
        assertEquals("", UnicodeTables.strip(nbsp));
        // and the JDK really does disagree about each of those four
        for (int cp : new int[] {0x0085, 0x00A0, 0x2007, 0x202F}) {
            assertTrue(UnicodeTables.isSpace(cp), "Python calls U+" + Integer.toHexString(cp)
                    + " whitespace");
            assertFalse(Character.isWhitespace(cp), "the JDK does not, which is the reason for"
                    + " the SPACE table");
        }
    }
}
