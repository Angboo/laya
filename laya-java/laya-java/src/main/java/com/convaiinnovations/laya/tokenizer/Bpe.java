package com.convaiinnovations.laya.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Byte-pair encoding over one pre-tokenized piece, ranked by the merge table.
 *
 * <p>This is the hot loop of the whole runtime and the easiest place for a port to be quietly
 * wrong, so it is written to one rule: <b>apply the lowest-ranked merge available, repeatedly,
 * until none applies.</b> Merging left-to-right instead, or applying every instance of one pair
 * before re-scanning, produces different ids for the same text on a minority of inputs and
 * identical ids on the majority -- the worst failure shape there is: a parity suite that passes on
 * prose and diverges on a customer's data.
 *
 * <p>Complexity: each pass over a piece of {@code k} symbols costs O(k) lookups and each merge
 * removes one symbol, so a piece costs O(k²) lookups worst case. Pieces are words, not documents,
 * so the quadratic term is bounded by the longest word. {@code laya-ts} hit exactly this shape and
 * fixed it upstream (5.5 s to 3 ms at 16k characters), which is why the per-pass scan stays over a
 * flat list rather than rebuilding structures per merge.
 */
final class Bpe {

    /**
     * Merge ranks, nested left-symbol -> right-symbol -> rank, rather than flat under a joined key.
     *
     * <p>A joined key ({@code left + " " + right}) is what the reference file format suggests, and
     * it is measurably safe for these two checkpoints -- all 630,613 merge entries across both were
     * checked and <b>no</b> merge symbol contains a space. But that is a property of two files, not
     * of the format: any separator character that can occur inside a symbol makes
     * {@code ("a b", "c")} and {@code ("a", "b c")} the same key, and the multilingual table is full
     * of symbols that are runs of whitespace and markers. The nested form cannot collide whatever
     * the symbols contain, and it allocates nothing per lookup, which the hot loop cares about more
     * than the extra map headers.
     */
    private final Map<String, Map<String, Integer>> merges;

    private final Map<String, Integer> vocab;
    private final String unknownToken;
    private final Integer unknownId;
    private final boolean fuseUnknown;
    private final boolean ignoreMerges;
    private final int[] byteFallbackIds;      // byte value -> id, or null when unavailable

    /**
     * @param vocab        symbol -> id, including the added-token ids
     * @param mergeRanks   merge pairs in file order; index is the rank, lower applies first
     * @param unknownToken the model's {@code unk_token}, or null when it declares none
     * @param fuseUnknown  the model's {@code fuse_unk}: collapse adjacent unknowns into one token
     * @param ignoreMerges the model's {@code ignore_merges}: a piece that is itself a vocabulary
     *                     entry is emitted whole, without running the merge loop
     * @param byteFallback the model's {@code byte_fallback}: resolve an unknown character to the
     *                     {@code <0xNN>} tokens of its UTF-8 bytes
     */
    Bpe(Map<String, Integer> vocab,
        List<String[]> mergeRanks,
        String unknownToken,
        boolean fuseUnknown,
        boolean ignoreMerges,
        boolean byteFallback) {
        this.vocab = vocab;
        this.unknownToken = unknownToken;
        this.unknownId = unknownToken == null ? null : vocab.get(unknownToken);
        this.fuseUnknown = fuseUnknown;
        this.ignoreMerges = ignoreMerges;
        if (unknownToken != null && this.unknownId == null) {
            throw new IllegalArgumentException(
                    "the tokenizer names " + unknownToken + " as its unknown token, but neither the "
                    + "vocabulary nor the added-token table has an entry for it");
        }

        this.merges = new HashMap<>(mergeRanks.size() * 2);
        for (int rank = 0; rank < mergeRanks.size(); rank++) {
            String[] pair = mergeRanks.get(rank);
            if (pair.length != 2) {
                throw new IllegalArgumentException(
                        "merge at rank " + rank + " has " + pair.length + " symbols, expected 2");
            }
            // Only the FIRST occurrence of a pair counts: the rank is the priority, and a later
            // duplicate must not demote an earlier, stronger merge.
            this.merges.computeIfAbsent(pair[0], key -> new HashMap<>(4))
                    .putIfAbsent(pair[1], rank);
        }

        this.byteFallbackIds = byteFallback ? resolveByteTokens(vocab) : null;
    }

    /**
     * The {@code <0xNN>} id for each byte value, or {@code -1} where the vocabulary has none.
     *
     * <p>All-or-nothing is <b>per character</b>, not per vocabulary. The multilingual checkpoint
     * ships <b>255</b> of the 256 byte tokens -- {@code <0x09>}, the tab, is absent -- and the
     * reference implementation resolves the bytes of one character and abandons the fallback for
     * <i>that character</i> only if one of its own bytes has no token. Treating a single missing
     * entry as "this model has no byte fallback" disables the whole mechanism: measured, it sent
     * 7,628 of 30,000 adversarial strings to {@code <unk>} where the reference emits byte tokens,
     * a 25.4% divergence that the 37-string fixture corpus did not contain a single case of.
     *
     * <p>(No character other than the tab itself encodes to a {@code 0x09} byte, and the tab is a
     * vocabulary entry in its own right, so the abandon path is unreachable for this checkpoint --
     * but reachability is the checkpoint's property, not this method's to assume.)
     */
    private static int[] resolveByteTokens(Map<String, Integer> vocab) {
        int[] ids = new int[256];
        for (int b = 0; b < 256; b++) {
            Integer id = vocab.get(String.format("<0x%02X>", b));
            ids[b] = id == null ? -1 : id;
        }
        return ids;
    }

    /** Token ids for one pre-tokenized piece, appended to {@code out}. */
    void encodePiece(String piece, List<Integer> out) {
        if (piece.isEmpty()) {
            return;
        }
        if (ignoreMerges) {
            // Only under the flag. The whole-piece lookup is an attractive unconditional
            // optimisation and it is not a safe one: it is exactly what `ignore_merges` turns on,
            // and both laya checkpoints declare it FALSE. The twelve vocabulary entries the merge
            // loop cannot rebuild in the multilingual checkpoint are all control tokens
            // (`<pad>`, `<eos>`, `<unk>`, `<mask>`, `<2mass>`, `[@BOS@]`, `<unused0..4>`), and
            // those are matched by the added-token stage before BPE ever sees them -- so taking the
            // shortcut anyway would be a divergence with no upside.
            Integer whole = vocab.get(piece);
            if (whole != null) {
                out.add(whole);
                return;
            }
        }

        List<String> symbols = initialSymbols(piece);
        mergeAll(symbols);
        emit(symbols, out);
    }

    /**
     * Applies merges to {@code symbols} in place, lowest rank first, re-scanning after each.
     *
     * <p>Re-scanning is not an optimisation choice: merging a pair creates a new symbol whose
     * neighbours may form a <i>lower</i>-ranked pair than anything that existed before, and a
     * single left-to-right pass would miss it.
     */
    private void mergeAll(List<String> symbols) {
        while (symbols.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            int bestIndex = -1;
            Map<String, Integer> fromLeft = merges.get(symbols.get(0));
            for (int i = 0; i < symbols.size() - 1; i++) {
                Map<String, Integer> current = fromLeft;
                // The next iteration's left symbol is this iteration's right symbol, so its row is
                // fetched once here and carried over instead of being looked up twice.
                fromLeft = merges.get(symbols.get(i + 1));
                if (current == null) {
                    continue;
                }
                Integer rank = current.get(symbols.get(i + 1));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0) {
                return;                                  // no merge applies: this piece is done
            }
            symbols.set(bestIndex, symbols.get(bestIndex) + symbols.get(bestIndex + 1));
            symbols.remove(bestIndex + 1);
        }
    }

    /**
     * Resolves merged symbols to ids: vocabulary entry, else the UTF-8 byte tokens, else unknown.
     *
     * <p>Resolving <i>after</i> merging rather than before is safe because every component of every
     * merge is itself a vocabulary entry -- {@link TokenizerModel} checks that at load time -- so a
     * symbol with no vocabulary entry cannot appear in any merge pair and therefore cannot have
     * merged with anything. That check is what keeps this ordering equivalent to the reference
     * implementation's, which classifies first and merges after.
     */
    private void emit(List<String> symbols, List<Integer> out) {
        boolean unknownPending = false;
        for (String symbol : symbols) {
            Integer id = vocab.get(symbol);
            if (id != null) {
                unknownPending = false;
                out.add(id);
                continue;
            }
            if (byteFallbackIds != null) {
                byte[] bytes = symbol.getBytes(StandardCharsets.UTF_8);
                boolean everyByteResolves = true;
                for (byte b : bytes) {
                    if (byteFallbackIds[b & 0xFF] < 0) {
                        everyByteResolves = false;
                        break;
                    }
                }
                // Per character: emit the byte tokens only if ALL of this character's bytes have
                // one, otherwise fall through to the unknown token for this character alone.
                if (everyByteResolves) {
                    for (byte b : bytes) {
                        out.add(byteFallbackIds[b & 0xFF]);
                    }
                    unknownPending = false;
                    continue;
                }
            }
            if (unknownId == null) {
                // Unreachable for a well-formed checkpoint: TokenizerModel refuses to load a model
                // that has neither an unknown token, nor a byte fallback, nor an alphabet covering
                // its pre-tokenizer's output. Dropping here rather than throwing keeps a damaged
                // vocabulary from taking down a request, and the load-time check is what makes the
                // case not silently lossy.
                continue;
            }
            if (fuseUnknown && unknownPending) {
                continue;                                // already emitted one for this run
            }
            out.add(unknownId);
            unknownPending = true;
        }
    }

    /**
     * The starting symbols: one per Unicode code point, not one per {@code char}.
     *
     * <p>Java strings are UTF-16, so an emoji or any astral character is two {@code char}s. Split
     * per {@code char} and the surrogate halves become symbols no vocabulary contains, so every
     * emoji would tokenize to unknown or vanish -- and prose would be unaffected, so a corpus
     * without astral characters would never show it.
     */
    private static List<String> initialSymbols(String piece) {
        List<String> symbols = new ArrayList<>(piece.length());
        int i = 0;
        while (i < piece.length()) {
            int codePoint = piece.codePointAt(i);
            int width = Character.charCount(codePoint);
            symbols.add(piece.substring(i, i + width));
            i += width;
        }
        return symbols;
    }

    /** The unknown token this model declares, or null when it declares none. */
    String unknownToken() {
        return unknownToken;
    }

    /** Whether this model can resolve any character through its UTF-8 bytes. */
    boolean hasByteFallback() {
        return byteFallbackIds != null;
    }
}
