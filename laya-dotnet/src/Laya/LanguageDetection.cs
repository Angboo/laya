using System.Globalization;
using System.Text;

namespace Laya;

/// <summary>
/// Dependency-free script and Latin-language detection used to route between Laya checkpoints.
/// Ports <c>laya/lang.py</c>.
/// </summary>
/// <remarks>
/// <para>
/// Routing only needs one decision: is this English Latin text, or is it something the English
/// checkpoint cannot read? Benchmarks on MASSIVE (14 languages) showed the English checkpoint
/// collapsing to near-random on non-Latin scripts, while holding up far better on Latin-script
/// languages. So the signal that matters most is script, and the secondary signal is whether
/// Latin text is English.
/// </para>
/// <para>
/// Script detection is exact. The Latin-script language guess is a stopword/diacritic heuristic
/// and is explicitly best-effort: pass an explicit model or language when it is already known.
/// </para>
/// <para>
/// Every method here iterates Unicode <em>code points</em> (via <see cref="Rune"/>), never UTF-16
/// code units, and applies Unicode-category checks rather than ASCII-only ones — matching
/// Python's <c>str.isalpha()</c>, <c>len()</c>, slicing and <c>str.lower()</c> semantics exactly.
/// A UTF-16-unit-based port would split astral characters (emoji, Deseret, Mathematical
/// Alphanumeric Symbols) into two units and disagree with Python on script counts and truncation
/// boundaries.
/// </para>
/// </remarks>
public static class LanguageDetection
{
    // Unicode blocks the English (ModernBERT-large, 50k English BPE) checkpoint cannot read.
    // Order matters: it decides which non-Latin script wins a tie in DetectScript (first
    // encountered in the source text wins, and Latin never wins a tie against a script that is
    // actually present, because "latin" is appended to the tally only after the scan finishes).
    private static readonly (string Name, (int Lo, int Hi)[] Ranges)[] ScriptRanges =
    [
        ("greek", [(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
        ("cyrillic", [(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)]),
        ("armenian", [(0x0530, 0x058F)]),
        ("hebrew", [(0x0590, 0x05FF)]),
        ("arabic", [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]),
        ("devanagari", [(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
        ("bengali", [(0x0980, 0x09FF)]),
        ("gurmukhi", [(0x0A00, 0x0A7F)]),
        ("gujarati", [(0x0A80, 0x0AFF)]),
        ("oriya", [(0x0B00, 0x0B7F)]),
        ("tamil", [(0x0B80, 0x0BFF)]),
        ("telugu", [(0x0C00, 0x0C7F)]),
        ("kannada", [(0x0C80, 0x0CFF)]),
        ("malayalam", [(0x0D00, 0x0D7F)]),
        ("sinhala", [(0x0D80, 0x0DFF)]),
        ("thai", [(0x0E00, 0x0E7F)]),
        ("lao", [(0x0E80, 0x0EFF)]),
        ("tibetan", [(0x0F00, 0x0FFF)]),
        ("myanmar", [(0x1000, 0x109F)]),
        ("georgian", [(0x10A0, 0x10FF)]),
        ("ethiopic", [(0x1200, 0x137F)]),
        ("khmer", [(0x1780, 0x17FF)]),
        ("hangul", [(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)]),
        ("kana", [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)]),
        ("han", [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)]),
    ];

    // Function words. Latin-script languages overlap heavily (de/la/le/un/e/que), so each hit is
    // weighted and a margin is required before calling something non-English. Order is load
    // bearing: LatinProfile's tie-break walks this list (excluding "en") and the first language
    // with the highest score wins.
    private static readonly string[] StopOrder = ["en", "fr", "de", "es", "pt", "it", "nl", "ro"];

    private static readonly Dictionary<string, HashSet<string>> Stop = new(StringComparer.Ordinal)
    {
        ["en"] = new(StringComparer.Ordinal)
        {
            "the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
            "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
            "will", "can", "would", "there", "their", "what", "which", "please", "we", "i",
        },
        ["fr"] = new(StringComparer.Ordinal)
        {
            "le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
            "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
            "et", "du", "au", "ou", "je", "tu", "il", "elle", "ils", "elles", "mon", "ton",
            "ma", "ta", "sa", "mes", "tes", "ses", "ces", "deux", "trois", "très", "bien",
            "tout", "tous", "toute", "fait", "veux", "veut", "peux", "peut", "dois", "doit",
            "merci", "bonjour", "jour", "jours", "mois", "fois", "quand", "comment", "pourquoi",
            "alors", "donc",
        },
        ["de"] = new(StringComparer.Ordinal)
        {
            "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
            "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
        },
        ["es"] = new(StringComparer.Ordinal)
        {
            "el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
            "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus",
            "la", "un", "y", "al", "lo", "le", "les", "su", "mi", "tu", "nos",
            "ni", "dos", "tres", "fue", "fueron", "ser", "tiene", "tienen", "tengo", "puede",
            "pueden", "quiero", "necesito", "hemos", "han", "sobre", "entre", "cuando", "donde",
            "porque", "aunque", "también", "ya", "eso", "esto", "esa", "ese", "nada", "algo",
            "aquí", "hoy", "gracias",
        },
        ["pt"] = new(StringComparer.Ordinal)
        {
            "os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
            "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela",
            "o", "e", "na", "nas", "nos", "ao", "aos", "por", "foi", "era", "ser", "sou",
            "tem", "tenho", "pode", "podem", "quero", "preciso", "eu", "meu", "minha", "seu",
            "sua", "isso", "isto", "aqui", "ali", "como", "quando", "onde", "porque", "mais",
            "já", "ainda", "agora", "hoje", "ontem", "dois", "três", "tudo", "nada", "obrigado",
            "olá",
        },
        ["it"] = new(StringComparer.Ordinal)
        {
            "il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
            "questo", "questa", "anche", "come", "più", "nella", "alla",
            "la", "le", "un", "uno", "una", "e", "ed", "o", "da", "su", "tra", "fra", "mi",
            "ci", "ne", "ho", "hai", "ha", "abbiamo", "avete", "hanno", "era", "stato", "stata",
            "devo", "deve", "devono", "voglio", "vorrei", "mio", "mia", "tuo", "sua", "quando",
            "dove", "perche", "molto", "poco", "sempre", "mai", "già", "ancora", "adesso", "oggi",
            "ieri", "grazie", "ciao", "scusa",
            "nel", "nell", "negli", "sul", "sulla", "sulle", "dal", "dalla", "dallo", "dagli", "dei",
            "delle", "dello", "degli", "agli", "alle", "col",
        },
        ["nl"] = new(StringComparer.Ordinal)
        {
            "het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
            "door", "maar", "ook", "worden", "deze", "naar", "wordt",
        },
        ["ro"] = new(StringComparer.Ordinal)
        {
            "și", "să", "este", "sunt", "care", "pentru", "din", "dar", "după", "până", "fără",
            "ale", "lui", "în", "fost", "acum", "vreau", "trebuie", "foarte", "acest", "această",
            "acesta", "aceasta", "mi", "ți", "vă", "nu",
        },
    };

    // Letters that ordinary English does not use. This is the signal that catches a Latin-script
    // language for which no stopword list is held at all (Romanian, Polish, Czech, Turkish,
    // Baltic, ...).
    private const string NonEnDiacriticChars =
        "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ"          // Western European
        + "ăâîșțşţ"                                 // Romanian
        + "ąćęłńśźż"                                // Polish
        + "čďěňřšťůž"                               // Czech / Slovak
        + "őű"                                      // Hungarian
        + "ğı"                                      // Turkish (text is lowercased before matching)
        + "āēģīķļņūž"                               // Baltic
        + "đ";                                      // Serbo-Croatian / Vietnamese

    private static readonly HashSet<int> NonEnDiacritics =
        new(NonEnDiacriticChars.EnumerateRunes().Select(r => r.Value));

    // Words that more than one list claims. Matching one says "not English" without saying
    // *which* language, so it alone may not name a winner (see LatinProfile).
    private static readonly HashSet<string> SharedWords = ComputeSharedWords();

    /// <summary>
    /// A diacritic rate above this is taken as evidence the text is not English, even when no
    /// stopword list matches it.
    /// </summary>
    private const double NonEnDiacriticRate = 0.02;

    private static HashSet<string> ComputeSharedWords()
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var set in Stop.Values)
            foreach (var w in set)
                counts[w] = counts.GetValueOrDefault(w) + 1;
        return new HashSet<string>(counts.Where(kv => kv.Value > 1).Select(kv => kv.Key), StringComparer.Ordinal);
    }

    /// <summary>
    /// Flatten a state into the text used for detection: string leaves of a string, dict-like or
    /// list-like value, joined with a single space and truncated to <paramref name="maxChars"/>
    /// Unicode code points. Keys are ignored, since they are usually English.
    /// </summary>
    /// <remarks>Recursion stops past 6 levels of nesting, matching Python's <c>_iter_text</c>.</remarks>
    public static string StateText(object? state, int maxChars = 4000)
    {
        var leaves = new List<string>();
        CollectText(state, leaves, 0);
        return SliceCodePoints(string.Join(" ", leaves), maxChars);
    }

    private static void CollectText(object? state, List<string> leaves, int depth)
    {
        if (depth > 6 || state is null) return;
        switch (state)
        {
            case string s:
                leaves.Add(s);
                return;
            // Dict-like before generically enumerable, mirroring PythonJson's dispatch order.
            case IEnumerable<KeyValuePair<string, object?>> kvpSeq:
                foreach (var kv in kvpSeq) CollectText(kv.Value, leaves, depth + 1);
                return;
            case System.Collections.IDictionary dict:
                foreach (System.Collections.DictionaryEntry entry in dict) CollectText(entry.Value, leaves, depth + 1);
                return;
            case System.Collections.IEnumerable seq:
                foreach (var item in seq) CollectText(item, leaves, depth + 1);
                return;
            default:
                // Numbers, booleans and other scalars contribute nothing, matching _iter_text.
                return;
        }
    }

    /// <summary>
    /// Dominant script of <paramref name="text"/>: <c>"latin"</c>, <c>"han"</c>,
    /// <c>"devanagari"</c>, ... or <c>"unknown"</c> if there are no letters.
    /// </summary>
    public static string DetectScript(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        // Non-Latin scripts are tallied in the order first encountered in the text; "latin" is
        // appended only after the scan, so it never wins a tie against a script actually present,
        // regardless of where in the text the Latin characters fall.
        var order = new List<string>();
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        var latin = 0;

        foreach (var rune in text.EnumerateRunes())
        {
            if (!Rune.IsLetter(rune)) continue;
            var cp = rune.Value;
            if (cp < 0x0250 || (cp >= 0x1E00 && cp <= 0x1EFF)) { latin++; continue; }
            foreach (var (name, ranges) in ScriptRanges)
            {
                if (!InRanges(cp, ranges)) continue;
                if (!counts.ContainsKey(name)) order.Add(name);
                counts[name] = counts.GetValueOrDefault(name) + 1;
                break;
            }
        }

        counts["latin"] = latin;
        order.Add("latin");

        var total = counts.Values.Sum();
        if (total == 0) return "unknown";

        var bestName = order[0];
        var bestCount = counts[bestName];
        for (var i = 1; i < order.Count; i++)
        {
            var name = order[i];
            if (counts[name] > bestCount) { bestName = name; bestCount = counts[name]; }
        }
        return bestName;
    }

    /// <summary>Fraction of alphabetic characters belonging to each detected script.</summary>
    public static IReadOnlyDictionary<string, double> ScriptProfile(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var counts = new Dictionary<string, int>(StringComparer.Ordinal) { ["latin"] = 0 };
        foreach (var rune in text.EnumerateRunes())
        {
            if (!Rune.IsLetter(rune)) continue;
            var cp = rune.Value;
            if (cp < 0x0250 || (cp >= 0x1E00 && cp <= 0x1EFF)) { counts["latin"]++; continue; }
            foreach (var (name, ranges) in ScriptRanges)
            {
                if (!InRanges(cp, ranges)) continue;
                counts[name] = counts.GetValueOrDefault(name) + 1;
                break;
            }
        }

        var total = counts.Values.Sum();
        if (total == 0) return new Dictionary<string, double>(0);

        var result = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var (k, v) in counts)
            if (v != 0) result[k] = (double)v / total;
        return result;
    }

    private static bool InRanges(int cp, (int Lo, int Hi)[] ranges)
    {
        foreach (var (lo, hi) in ranges)
            if (cp >= lo && cp <= hi) return true;
        return false;
    }

    /// <summary>
    /// Evidence behind the Latin-script language guess: <see cref="Language"/> (may be
    /// <see langword="null"/> when undecided), <see cref="EnglishHits"/>,
    /// <see cref="DiacriticRate"/> and <see cref="LooksNonEnglish"/>. <see cref="LanguageDetection.Analyse"/>
    /// needs the evidence and not just the verdict, because "undecided" and "English" are
    /// different answers and only one of them is safe to send to the English checkpoint.
    /// </summary>
    /// <param name="Language">
    /// The best-effort language code, or <see langword="null"/> when undecided. Only named when
    /// it matched at least one word that no other list claims: shared function words alone
    /// (<c>la</c>, <c>e</c>, <c>o</c>) identify no particular language.
    /// </param>
    /// <param name="EnglishHits">How many words matched the English stopword list.</param>
    /// <param name="DiacriticRate">
    /// Fraction of (lowercased) code points that are letters ordinary English does not use.
    /// Not rounded; <see cref="LanguageAnalysis.DiacriticRate"/> rounds this to 4 decimals.
    /// </param>
    /// <param name="LooksNonEnglish"><see cref="DiacriticRate"/> at or above the detection threshold.</param>
    public readonly record struct LatinProfileResult(
        string? Language, int EnglishHits, double DiacriticRate, bool LooksNonEnglish);

    /// <summary>
    /// Evidence behind the Latin-script language guess. See <see cref="LatinProfileResult"/>.
    /// </summary>
    public static LatinProfileResult LatinProfile(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var words = ExtractWords(text).Select(PythonLower).ToList();
        var lowered = PythonLower(text);

        var diac = 0;
        var loweredLen = 0;
        foreach (var rune in lowered.EnumerateRunes())
        {
            loweredLen++;
            if (NonEnDiacritics.Contains(rune.Value)) diac++;
        }
        var diacRate = (double)diac / Math.Max(1, loweredLen);
        var nonEnglish = diacRate >= NonEnDiacriticRate;

        if (words.Count < 4)
            return new LatinProfileResult(null, 0, diacRate, nonEnglish);

        var wordSet = new HashSet<string>(words, StringComparer.Ordinal);
        var scores = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var lang in StopOrder)
            scores[lang] = words.Count(w => Stop[lang].Contains(w));
        var en = scores.GetValueOrDefault("en");

        // Only a language that matched at least one word no other list claims may be named.
        // Without that condition the top score can be pure overlap. Such a language is dropped
        // entirely from the running rather than merely losing the tie, so a lesser score with
        // real evidence still gets named, and the text stays undecided when no list has any.
        string? bestLg = null;
        var best = 0;
        foreach (var lang in StopOrder)
        {
            if (lang == "en") continue;
            var evidenced = false;
            foreach (var w in wordSet)
                if (Stop[lang].Contains(w) && !SharedWords.Contains(w)) { evidenced = true; break; }
            if (!evidenced) continue;

            var score = scores[lang];
            if (bestLg is null || score > best) { bestLg = lang; best = score; }
        }

        string? language = null;
        if (bestLg is not null && best >= Math.Max(2, en + 2))
        {
            // a non-English language needs a clear margin over English function words
            language = bestLg;
        }
        else if (bestLg is not null && nonEnglish && best >= Math.Max(2, en))
        {
            // Needs two hits here too: one shared function word on the strength of the
            // diacritics alone is a guess dressed as a detection.
            language = bestLg;
        }
        else if (en > 0 && !nonEnglish)
        {
            language = "en";
        }

        return new LatinProfileResult(language, en, diacRate, nonEnglish);
    }

    /// <summary>
    /// Best-effort language code for Latin-script text, or <see langword="null"/> when
    /// undecided.
    /// </summary>
    /// <remarks>
    /// Scores function-word hits per language and requires the winner to beat English by a
    /// margin and to have matched at least one word of its own, so ordinary English is never
    /// misrouted and a word of several languages at once names none of them. Short inputs
    /// usually return <see langword="null"/> on purpose.
    /// </remarks>
    public static string? GuessLatinLanguage(string text) => LatinProfile(text).Language;

    /// <summary>Full detection result for a state.</summary>
    public static LanguageAnalysis Analyse(object? state)
    {
        var text = StateText(state);
        var prof = ScriptProfile(text);
        var script = DetectScript(text);
        var nonLatin = prof.Count > 0 ? Calibration.Round4(1.0 - prof.GetValueOrDefault("latin", 0.0)) : 0.0;

        if (script == "unknown")
            return new LanguageAnalysis("unknown", prof, null, true, true, 0.0, 0.0);

        if (script != "latin")
            return new LanguageAnalysis(script, prof, null, false, true, 0.0, nonLatin);

        var profLat = LatinProfile(text);
        var lang = profLat.Language;
        // Undecided is not English. Treating it as English sends every Latin-script language no
        // stopwords are held for to the checkpoint that cannot read it, silently.
        var undecided = lang is null;
        var english = lang == "en" || (undecided && !profLat.LooksNonEnglish);
        return new LanguageAnalysis("latin", prof, lang, english, undecided,
            Calibration.Round4(profLat.DiacriticRate), nonLatin);
    }

    /// <summary>Whether the English checkpoint can be expected to read this state.</summary>
    public static bool IsEnglish(object? state) => Analyse(state).IsEnglish;

    // ── word extraction ──────────────────────────────────────────────────────

    /// <summary>
    /// Runs of code points matching Python's <c>[^\W\d_]+</c>: word characters (letters, plus
    /// non-decimal numeric characters such as Roman numerals) excluding decimal digits and the
    /// underscore.
    /// </summary>
    private static List<string> ExtractWords(string text)
    {
        var words = new List<string>();
        var current = new StringBuilder();
        foreach (var rune in text.EnumerateRunes())
        {
            if (IsWordRune(rune))
            {
                current.Append(rune);
            }
            else if (current.Length > 0)
            {
                words.Add(current.ToString());
                current.Clear();
            }
        }
        if (current.Length > 0) words.Add(current.ToString());
        return words;
    }

    private static bool IsWordRune(Rune r)
    {
        if (Rune.IsLetter(r)) return true;
        var category = Rune.GetUnicodeCategory(r);
        // Python's \w = str.isalnum() (letters, plus Nd/Nl/No numeric categories) or "_"; \d
        // matches only decimal digits (Nd). [^\W\d_]+ therefore keeps letters and the
        // non-decimal numeric categories (Nl, No — e.g. Roman numerals, vulgar fractions) but
        // excludes Nd and the literal underscore.
        return category is UnicodeCategory.LetterNumber or UnicodeCategory.OtherNumber;
    }

    // ── code-point-aware slicing ─────────────────────────────────────────────

    /// <summary>
    /// Python's <c>str.lower()</c>: full (locale-independent default) Unicode lowercasing,
    /// including the one unconditional 1-to-2-code-point special case, U+0130 (LATIN CAPITAL
    /// LETTER I WITH DOT ABOVE) → U+0069 U+0307 ("i" + combining dot above).
    /// </summary>
    /// <remarks>
    /// This project builds with <c>InvariantGlobalization</c> enabled, under which
    /// <see cref="string.ToLowerInvariant()"/> uses a compiled-in simple (1-to-1) Unicode casing
    /// table rather than ICU's full <c>SpecialCasing.txt</c> rules: it lowercases ordinary
    /// accented letters correctly (é, ß, ...) but leaves U+0130 completely untouched rather than
    /// expanding it, so the fix-up below only ever needs to look for a leftover U+0130.
    /// </remarks>
    internal static string PythonLower(string text)
    {
        var lowered = text.ToLowerInvariant();
        return lowered.Contains('İ') ? lowered.Replace("İ", "i̇") : lowered;
    }

    /// <summary>Python's <c>s[:maxCodePoints]</c>: a prefix measured in Unicode code points.</summary>
    /// <remarks>Internal rather than private so <see cref="LayaEmail"/> can reuse it for <c>max_chars</c>.</remarks>
    internal static string SliceCodePoints(string text, int maxCodePoints)
    {
        if (maxCodePoints <= 0) return string.Empty;
        var sb = new StringBuilder();
        var count = 0;
        foreach (var rune in text.EnumerateRunes())
        {
            if (count >= maxCodePoints) break;
            sb.Append(rune);
            count++;
        }
        return sb.ToString();
    }
}

/// <summary>
/// Full language/script detection result for a state. Every field is populated on every branch
/// (script <c>"unknown"</c>, non-Latin, or Latin), so a caller never needs to guard on which
/// branch produced it.
/// </summary>
/// <param name="Script">
/// <c>"unknown"</c> (no letters), <c>"latin"</c>, or a non-Latin script name such as
/// <c>"devanagari"</c> or <c>"han"</c>.
/// </param>
/// <param name="ScriptProfile">Fraction of alphabetic characters belonging to each detected script.</param>
/// <param name="Language">Best-effort ISO-ish language code for Latin-script text, or <see langword="null"/>.</param>
/// <param name="IsEnglish">Whether the English checkpoint can be expected to read this state.</param>
/// <param name="LanguageUndecided">
/// True when the script is not Latin (language is not applicable), or the script is Latin but no
/// language could be identified.
/// </param>
/// <param name="DiacriticRate">
/// Fraction of lowercased code points that are letters ordinary English does not use. Rounded to
/// 4 decimals. Always 0 outside the Latin-script branch.
/// </param>
/// <param name="NonLatinFraction">
/// <c>1 - (fraction of alphabetic characters that are Latin)</c>, rounded to 4 decimals; 0 when
/// there are no letters at all.
/// </param>
public sealed record LanguageAnalysis(
    string Script,
    IReadOnlyDictionary<string, double> ScriptProfile,
    string? Language,
    bool IsEnglish,
    bool LanguageUndecided,
    double DiacriticRate,
    double NonLatinFraction);
