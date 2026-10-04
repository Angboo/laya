package com.convaiinnovations.laya.decode;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.convaiinnovations.laya.Fixtures;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Every number in an answer goes through {@code round(value, 4)}, and Python rounds half to EVEN
 * on the exact binary value.
 *
 * <p>The fixture separates real softmax output from exactly-representable halfway values, because
 * the two groups say different things: on softmax output a half-up implementation also agrees, so
 * only the halfway group can tell the two rules apart. A failure names which group broke.
 */
final class RoundingTest {

    @Test
    @DisplayName("half-even matches CPython on every group, including engineered halfway values")
    @SuppressWarnings("unchecked")
    void matchesCPython() {
        Map<String, Object> groups = (Map<String, Object>) Fixtures.load("python_json.json").get("round4");
        assertTrue(groups.containsKey("halfway"), "the halfway group is what discriminates the rules");
        for (Map.Entry<String, Object> group : groups.entrySet()) {
            for (Object entry : (List<Object>) group.getValue()) {
                Map<String, Object> row = (Map<String, Object>) entry;
                double value = Double.longBitsToDouble(((Number) row.get("bits")).longValue());
                assertEquals(((Number) row.get("r4")).doubleValue(), Rounding.round4(value),
                        "round4 of bits " + row.get("bits") + " in group " + group.getKey());
            }
        }
    }

    @Test
    @DisplayName("the laya-ts spelling agrees on softmax output and disagrees on halfway values")
    @SuppressWarnings("unchecked")
    void halfUpAgreesOnlyWhereItCan() {
        Map<String, Object> groups = (Map<String, Object>) Fixtures.load("python_json.json").get("round4");
        int softmaxMismatches = countMismatches((List<Object>) groups.get("softmax"));
        int halfwayMismatches = countMismatches((List<Object>) groups.get("halfway"));
        // This is the measurement that justified departing from `laya-ts` here, kept as a test so
        // the claim cannot rot into a comment that is no longer true.
        assertEquals(0, softmaxMismatches,
                "half-up should match CPython on real softmax output");
        assertTrue(halfwayMismatches > 0,
                "the halfway group must actually discriminate the two rounding rules");
    }

    @SuppressWarnings("unchecked")
    private static int countMismatches(List<Object> rows) {
        int mismatches = 0;
        for (Object entry : rows) {
            Map<String, Object> row = (Map<String, Object>) entry;
            double value = Double.longBitsToDouble(((Number) row.get("bits")).longValue());
            if (Rounding.round4HalfUp(value) != ((Number) row.get("r4")).doubleValue()) {
                mismatches++;
            }
        }
        return mismatches;
    }

    @Test
    @DisplayName("a non-finite value passes through rather than raising")
    void nonFinite() {
        assertTrue(Double.isNaN(Rounding.round4(Double.NaN)));
        assertEquals(Double.POSITIVE_INFINITY, Rounding.round4(Double.POSITIVE_INFINITY));
    }
}
