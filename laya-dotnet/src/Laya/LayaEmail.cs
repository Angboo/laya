using System.Text.RegularExpressions;

namespace Laya;

/// <summary>Email utilities for cleaning and structuring email inputs. Ports <c>laya/email.py</c>.</summary>
public static partial class LayaEmail
{
    // re.match anchors at the start only; every pattern below carries its own leading `^`, so
    // .NET's Regex.IsMatch (which is not implicitly start-anchored) reproduces that exactly.
    // `re.I` -> RegexOptions.IgnoreCase.
    [GeneratedRegex(@"^\s*On .{0,300}wrote:\s*$", RegexOptions.IgnoreCase)]
    private static partial Regex QuoteHeaderOnWrote();

    [GeneratedRegex(@"^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}", RegexOptions.IgnoreCase)]
    private static partial Regex QuoteHeaderOriginalOrForwarded();

    [GeneratedRegex(@"^\s*_{8,}\s*$")]
    private static partial Regex QuoteHeaderUnderscores();

    [GeneratedRegex(@"^\s*From:\s.+$", RegexOptions.IgnoreCase)]
    private static partial Regex QuoteHeaderFromLine();

    [GeneratedRegex(@"^\s*--\s*$")]
    private static partial Regex SignatureDashDash();

    [GeneratedRegex(@"^\s*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[\w ,!.]*$",
        RegexOptions.IgnoreCase)]
    private static partial Regex SignatureClosing();

    [GeneratedRegex(@"^\s*sent from my (iphone|android|mobile|ipad)", RegexOptions.IgnoreCase)]
    private static partial Regex SignatureSentFromMobile();

    // .search() semantics: no anchors, matches anywhere in the text.
    [GeneratedRegex(
        @"(confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|"
        + @"if you (have )?received this (e-?mail|message) in error)",
        RegexOptions.IgnoreCase)]
    private static partial Regex Disclaimer();

    // Sentence boundary: split just after a `.`/`!`/`?` and the whitespace that follows it. .NET
    // Regex supports the lookbehind natively, so this needs no hand-written splitter.
    [GeneratedRegex(@"(?<=[.!?])\s+")]
    private static partial Regex SentenceBoundary();

    [GeneratedRegex(@"\n\s*\n")]
    private static partial Regex ParagraphBreak();

    [GeneratedRegex(@"[ \t]+")]
    private static partial Regex SpaceTabRun();

    private static readonly Regex[] QuoteHeaders =
    [
        QuoteHeaderOnWrote(), QuoteHeaderOriginalOrForwarded(), QuoteHeaderUnderscores(), QuoteHeaderFromLine(),
    ];

    private static readonly Regex[] SignatureMarkers =
    [
        SignatureDashDash(), SignatureClosing(), SignatureSentFromMobile(),
    ];

    /// <summary>Remove quoted email history, signatures and disclaimers to keep input focused.</summary>
    public static string CleanBody(string? body, int maxChars = 3000)
    {
        var text = (body ?? string.Empty).Replace("\r\n", "\n").Replace("\r", "\n").Replace("\\n", "\n");

        var lines = new List<string>();
        foreach (var line in text.Split('\n'))
        {
            if (lines.Count > 0 && Array.Exists(QuoteHeaders, p => p.IsMatch(line))) break;
            if (line.TrimStart().StartsWith('>')) continue;
            lines.Add(line.TrimEnd());
        }

        var cut = lines.Count;
        var start = Math.Max(1, Math.Min((int)(lines.Count * 0.6), lines.Count - 8));
        for (var i = start; i < lines.Count; i++)
        {
            var stripped = lines[i].Trim();
            if (CodePointLength(stripped) <= 40 && Array.Exists(SignatureMarkers, p => p.IsMatch(lines[i])))
            {
                cut = i;
                break;
            }
        }
        lines = lines.GetRange(0, cut);

        var paragraphs = ParagraphBreak().Split(string.Join('\n', lines)).Select(StripDisclaimer);
        var collapsed = string.Join("\n\n", paragraphs.Select(p => p.Trim()).Where(p => p.Length > 0));
        text = SpaceTabRun().Replace(collapsed, " ");
        return LanguageDetection.SliceCodePoints(text, maxChars);
    }

    /// <summary>
    /// Drop boilerplate disclaimer text from one paragraph.
    /// </summary>
    /// <remarks>
    /// A paragraph is dropped whole only when *every* sentence in it is boilerplate; otherwise
    /// only the boilerplate sentences go. A footer that runs on without a blank line would
    /// otherwise take the sender's actual request with it, which is worse than leaving one
    /// boilerplate line behind.
    /// </remarks>
    private static string StripDisclaimer(string paragraph)
    {
        if (!Disclaimer().IsMatch(paragraph)) return paragraph; // nothing to do: keep the original line structure
        var parts = SentenceBoundary().Split(paragraph).Select(p => p.Trim()).Where(p => p.Length > 0);
        return string.Join(' ', parts.Where(p => !Disclaimer().IsMatch(p)));
    }

    /// <summary>Construct a clean state dictionary for email classification.</summary>
    /// <param name="subject">The email subject. Whitespace-trimmed regardless of <paramref name="clean"/>.</param>
    /// <param name="body">The email body.</param>
    /// <param name="sender">Optional sender, stored under the key <c>"from"</c> when non-empty.</param>
    /// <param name="clean">
    /// Whether <paramref name="body"/> is run through <see cref="CleanBody"/>. When
    /// <see langword="false"/>, the raw body is kept as-is.
    /// </param>
    /// <param name="extra">
    /// Additional fields to merge in, in the given order; a <see langword="null"/> value is
    /// dropped entirely rather than stored, and a key that collides with <c>subject</c>/<c>body</c>/
    /// <c>from</c> overwrites it in place, matching Python's <c>**extra</c> plus <c>dict.update</c>.
    /// </param>
    public static IReadOnlyDictionary<string, object?> State(
        string? subject, string? body, string? sender = null, bool clean = true,
        IEnumerable<KeyValuePair<string, object?>>? extra = null)
    {
        var state = new OrderedDictionary<string, object?>(StringComparer.Ordinal)
        {
            ["subject"] = (subject ?? string.Empty).Trim(),
            ["body"] = clean ? CleanBody(body) : body ?? string.Empty,
        };
        if (!string.IsNullOrEmpty(sender)) state["from"] = sender;
        if (extra is not null)
            foreach (var (key, value) in extra)
                if (value is not null) state[key] = value;
        return state;
    }

    private static int CodePointLength(string s)
    {
        var count = 0;
        foreach (var _ in s.EnumerateRunes()) count++;
        return count;
    }
}
