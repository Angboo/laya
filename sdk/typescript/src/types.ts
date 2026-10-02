/** Values that survive the JSON transport without being changed or discarded. */
export type JsonValue = string | number | boolean | null | readonly JsonValue[] | { readonly [key: string]: JsonValue };
export type State = string | readonly JsonValue[] | { readonly [key: string]: JsonValue } | null;

export interface ChoiceQuestion {
  readonly type: 'choice';
  readonly instructions: JsonValue;
  readonly criteria: readonly string[] | Readonly<Record<string, JsonValue>>;
}
export interface ScoreQuestion {
  readonly type: 'score';
  readonly instructions: JsonValue;
  readonly criteria: readonly JsonValue[];
}
export interface NoulQuestion {
  readonly type: 'noul';
  readonly instructions: JsonValue;
  readonly criteria?: Readonly<{ true?: JsonValue; false?: JsonValue }> | null;
}
export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;
export type Questions = Readonly<Record<string, Question>>;
export type ModelName = 'english' | 'multilingual' | 'typed-decisions';
export type ModelAlias = ModelName | 'en' | 'laya' | 'default' | 'multi' | 'ml' | 'laya-multilingual'
  | 'typed' | 'typed_decisions' | 'laya-typed-decisions' | 'decisions';

export interface RequestOptions {
  /** Abort waiting for the response. Running inference may still finish on the server. */
  signal?: AbortSignal;
  /** Override the client timeout. Zero disables the timeout. */
  timeoutMs?: number;
}
export interface PredictOptions extends RequestOptions {
  /** Local Laya checkpoint or alias. Overrides the client's configured model. */
  model?: string;
  /** Force a checkpoint by workflow name instead of letting routing decide; the server
   *  answers 422 naming an unknown one. */
  task?: string;
  /** A language code (`de`, `en-US`) that skips detection when it names a language. */
  lang?: string;
  /** A language code from the client's own identifier, consulted after `lang` and
   *  before detection. */
  langGuess?: string;
  /** Total token window for this request, capped by the deployment's
   *  `LAYA_MAX_TOKEN_BUDGET` (422 above it). */
  maxLen?: number;
  /** Token window the option prompt shares, same cap. */
  headMaxLen?: number;
  /** Abstention threshold in `[0, 1]`; an answer whose `answer_confidence` falls below
   *  it comes back marked `low_confidence`, with the answer itself kept. */
  minConfidence?: number;
}

interface AnswerBase {
  /** Optional action metadata returned by Laya. */
  action?: { act_probability: number };
}
export interface ChoiceAnswer<Label extends string = string> extends AnswerBase {
  type: 'choice';
  confidence: number;
  choice: Label;
  probabilities: Record<Label, number>;
}
export interface ScoreAnswer extends AnswerBase {
  type: 'score';
  confidence: number;
  /** Expected zero-based rubric index, not a normalized probability. */
  score: number;
  legend: Record<string, JsonValue>;
  probabilities: Record<string, number>;
}
export interface NoulAnswer extends AnswerBase {
  type: 'noul';
  /** Optional confidence metadata returned by Laya. */
  confidence?: number;
  /** P(true), between 0 and 1. */
  noul: number;
}
type Labels<C> = C extends readonly string[] ? C[number] : Extract<keyof C, string>;
export type ChoiceLabels<Q extends ChoiceQuestion> = Labels<Q['criteria']>;
export type Answer<Q extends Question = Question> = Q extends ChoiceQuestion
  ? ChoiceAnswer<ChoiceLabels<Q>> : Q extends ScoreQuestion ? ScoreAnswer : NoulAnswer;
export type Answers<Q extends Questions> = { -readonly [K in keyof Q]: Answer<Q[K]> };

export interface LanguageDetection {
  script: string;
  script_profile: Record<string, number>;
  language: string | null;
  is_english: boolean;
  language_undecided: boolean;
  diacritic_rate: number;
  non_latin_fraction: number;
}
export interface RouteDecision {
  model: ModelName;
  repo: string;
  reason: string;
  detection: LanguageDetection | null;
  workflow: string | null;
}
/** One question whose options no longer have a token span each after the head budget (#538). */
export interface OptionCollapse {
  /** Options the question defines, not the option markers that reached the sequence. */
  total: number;
  /** Of those, the ones still carrying a span of their own. */
  distinct: number;
  /** Tokens the budget allowed each option, or null when none was capped. */
  tokens_per_option: number | null;
}
/** What the forward pass was built from, as `/v1/systemone` reports it.
 *
 *  How much of a state the model reads is a token budget, not a character count, and the budget
 *  moves with `max_len`, `head_max_len` and each question's own option prompt (#174). A truncated
 *  answer is still an answer, so these keys are the only place the cut is visible: nothing in
 *  `answers` changes when evidence is dropped.
 */
export interface Usage {
  /** Non-pad tokens of the state's rows, one row per question, so it grows with the questions. */
  input_tokens: number;
  /** Always 0: the head answers in one pass and generates nothing. */
  output_tokens: number;
  /** Tokens the whole serialized state needs. */
  state_tokens: number;
  /** Tokens of it at least one question did not get: the worst case over the questions, since
   *  each leaves the state a different room. */
  state_tokens_dropped: number;
  /** Whether that worst case dropped anything. */
  truncated: boolean;
  /** The ids whose own window was cut, empty when none was. */
  truncated_questions: string[];
  /** Present only when some question's options were collapsed, keyed by question id. */
  options?: Record<string, OptionCollapse>;
}
export interface Prediction<Q extends Questions = Questions> {
  model: string;
  answers: Answers<Q>;
  usage: Usage;
  /** Optional routing metadata returned by Laya. */
  routing?: RouteDecision;
}
/** Laya's self-hosted health probe.
 *
 * `status` is always present. The rest is only answered to a caller the server can authenticate:
 * on a deployment that set `LAYA_API_KEY`, an anonymous probe gets `{status: 'ok'}` and nothing
 * else, because the other fields name resident checkpoints, their revision SHAs and the host's
 * device state. A deployment with no key set answers every field to everyone.
 */
export interface Health {
  status: 'ok';
  loaded?: ModelName[];
  device?: string;
}

/** Preserve question IDs, primitive types, and literal choice labels when declaring a schema. */
export function defineQuestions<const Q extends Questions>(questions: Q): Q {
  return questions;
}
