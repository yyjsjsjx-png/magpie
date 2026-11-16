package gateway

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/provider"
)

// A fork of a conversation at its last finished turn goes on in a branch of
// the lead's session, cut after the reply it follows. DeepSeek Harness's
// subagent_fork starts a sub-agent on the lead's finished turns and a task
// of its own, while the lead's run answers the lead's next turn: the fork's
// last reply calls no tool, so branch found no call line, and the fork is
// no retry of the busy turn, so retryBranch took none either. A new Claude
// Code was told the whole conversation, and on 11-17 a fork whose prefix
// was 135,426 tokens read 21,671 (Claude Code's system prompt and tools)
// and wrote 113,755 to the cache.
func TestClaudeForkWhileItsLeadIsBusyBranchesAfterTheReply(t *testing.T) {
	for _, mode := range []string{"print", "interactive"} {
		t.Run(mode, func(t *testing.T) { forkWhileLeadBusy(t, mode) })
	}
}

func forkWhileLeadBusy(t *testing.T, mode string) {
	h := newBusyRetryHarness(t, mode)
	pickFiveMinutes(t)
	logs := keepLog(t)
	task := rmsg("user", "the conversation so far, told at length. fix the bug")
	first := h.ask(`[` + task + `]`)
	if !strings.Contains(first, "turn 1") {
		t.Fatalf("the first turn: %q", first)
	}
	reply := rmsg("assistant", first)

	// the lead's next turn: its run holds mid-reply, busy
	firstTold := h.heard()
	done := h.send(`[` + task + `,` + reply + `,` + rmsg("user", "and run the tests") + `]`)
	t.Cleanup(func() {
		h.s.subscription.abortAll() // the lead's turn holds to the end: stop it
		<-done
	})
	deadline := time.Now().Add(10 * time.Second)
	for !strings.Contains(h.read(), "and run the tests") || h.heard() == firstTold {
		if time.Now().After(deadline) {
			t.Fatalf("the lead's run was never told its turn:\n%s\nmagpie log:\n%s", h.read(), logs.String())
		}
		time.Sleep(10 * time.Millisecond)
	}

	// the fork: the lead's finished turns, and a task of its own
	if got := h.ask(`[` + task + `,` + reply + `,` + rmsg("user", "update the docs for the fix") + `]`); got == "" {
		t.Fatal("the fork was answered nothing")
	}
	branch, lead := 0, 0
	for name, told := range h.toldBy() {
		if !strings.Contains(told, "--resume") {
			lead++
			continue
		}
		branch++
		if strings.Contains(told, "the conversation so far") {
			t.Errorf("the fork's run %s was told the whole conversation:\n%.600s", name, told)
		}
		if !strings.Contains(told, "update the docs for the fix") || strings.Contains(told, "and run the tests") {
			t.Errorf("the fork's run %s was told, want its task and not the lead's turn:\n%.600s", name, told)
		}
		// a copy of the lead's session (lead<pid>), never the lead's own
		i := strings.Index(told, "--resume ")
		if sid := told[i+9:]; strings.HasPrefix(sid, "lead") {
			t.Errorf("the fork's run resumed the lead's session %.30s, want a copy of its own", sid)
		}
	}
	if branch != 1 || lead != 1 {
		t.Fatalf("%d runs branched, %d started anew, want one of each; told log:\n%s\nmagpie log:\n%s", branch, lead, h.read(), logs.String())
	}
	if want := "cut after that reply"; !strings.Contains(logs.String(), want) {
		t.Errorf("the log says:\n%s", logs.String())
	}
	for _, whole := range []string{"a new Claude Code is told the whole conversation: 3 messages", "from a transcript of it"} {
		if strings.Contains(logs.String(), whole) {
			t.Errorf("the fork told a new Claude Code the whole conversation:\n%s", logs.String())
		}
	}
	// the branch keeps the 5m cache the lead's run had
	everyRunKept5m(t, h.logs)
}

// A turn sent in place of one answered two turns back goes on in a branch
// of the run's session, cut after the reply before it: the user went back
// in the conversation, rewrote a message, and sent it, while the run waits
// after its latest reply. A rewrite of the latest turn is taken back to
// before it in its run (behindKey, #1638); one further back found no run,
// and a new Claude Code was told the whole conversation (on 11-14 a
// prefix of 318,991 tokens read 21,682 and wrote 297,307). The run keeps
// waiting after its latest reply.
func TestClaudeTurnSentInPlaceOfAnEarlierOneBranchesAfterTheReplyBefore(t *testing.T) {
	for _, mode := range []string{"print", "interactive"} {
		t.Run(mode, func(t *testing.T) { earlierTurnSentAnew(t, mode) })
	}
}

func earlierTurnSentAnew(t *testing.T, mode string) {
	h := newBusyRetryHarness(t, mode)
	t.Cleanup(h.s.subscription.abortAll)
	if err := os.WriteFile(filepath.Join(filepath.Dir(h.told), "nohold"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	pickFiveMinutes(t)
	logs := keepLog(t)
	task := rmsg("user", "the conversation so far, told at length. fix the bug")
	first := h.ask(`[` + task + `]`)
	conv := `[` + task + `,` + rmsg("assistant", first)
	second := h.ask(conv + `,` + rmsg("user", "and run the tests") + `]`)
	third := h.ask(conv + `,` + rmsg("user", "and run the tests") + `,` + rmsg("assistant", second) + `,` + rmsg("user", "now commit it") + `]`)
	if !strings.Contains(third, "turn 3") || pidOf(t, third) != pidOf(t, first) {
		t.Fatalf("the conversation's turns went to more than one run: %q, %q, %q\nmagpie log:\n%s", first, second, third, logs.String())
	}

	// the second turn, rewritten, in place of the second and third
	if got := h.ask(conv + `,` + rmsg("user", "run only the unit tests") + `]`); got == "" {
		t.Fatal("the rewritten turn was answered nothing")
	}
	branch := 0
	for name, told := range h.toldBy() {
		if !strings.Contains(told, "--resume") {
			continue
		}
		branch++
		if strings.Contains(told, "the conversation so far") {
			t.Errorf("the rewritten turn's run %s was told the whole conversation:\n%.600s", name, told)
		}
		if !strings.Contains(told, "run only the unit tests") || strings.Contains(told, "and run the tests") || strings.Contains(told, "now commit it") {
			t.Errorf("the rewritten turn's run %s was told, want the rewritten turn alone:\n%.600s", name, told)
		}
	}
	if branch != 1 {
		t.Fatalf("%d runs branched, want one; told log:\n%s\nmagpie log:\n%s", branch, h.read(), logs.String())
	}
	if want := "cut after that reply"; !strings.Contains(logs.String(), want) {
		t.Errorf("the log says:\n%s", logs.String())
	}
	for _, whole := range []string{"a new Claude Code is told the whole conversation: 3 messages", "from a transcript of it"} {
		if strings.Contains(logs.String(), whole) {
			t.Errorf("the rewritten turn told a new Claude Code the whole conversation:\n%s", logs.String())
		}
	}
	// the run still waits after its latest reply, for the turn after it
	h.s.subscription.mu.Lock()
	idle := len(h.s.subscription.idle)
	h.s.subscription.mu.Unlock()
	if idle != 2 {
		t.Errorf("%d runs wait, want the conversation's and the branch's", idle)
	}
	everyRunKept5m(t, h.logs)
}

// A branch's copy holds the run's session as far as the reply the turn
// follows, and no further: a fork at the first reply of four, and one at
// the second, each copy the turns up to their reply alone, and the run
// they branched off goes on with the conversation's own next turn. The
// stand-in Claude Code resumes any session file it finds, so what the new
// turn was told says nothing of where the copy was cut; the copy itself
// is read, and held to the run's session as it stood after that reply,
// byte for byte, with the branch's own turn after it and nothing else: a
// stand-in started anew numbers its lines from 1 again, so its own line-1
// would pass for the run's first reply in a copy cut before it. Reviews
// that pointed every turn start at the run's latest reply, and one that
// cut the first start before its reply, found the tests before this green.
func TestClaudeTurnBranchCopiesTheSessionAsFarAsTheReply(t *testing.T) {
	for _, mode := range []string{"print", "interactive"} {
		t.Run(mode, func(t *testing.T) { copiesAsFarAsTheReply(t, mode) })
	}
}

func copiesAsFarAsTheReply(t *testing.T, mode string) {
	h := newBusyRetryHarness(t, mode)
	t.Cleanup(h.s.subscription.abortAll)
	if err := os.WriteFile(filepath.Join(filepath.Dir(h.told), "nohold"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	// session reads the file of the session sid, with its id taken out
	session := func(sid string) string {
		t.Helper()
		var dirs []string
		h.s.subscription.mu.Lock()
		for _, run := range h.s.subscription.runs {
			run.mu.Lock()
			if run.sessionID == sid {
				dirs = slices.Clone(run.sessions)
			}
			run.mu.Unlock()
		}
		h.s.subscription.mu.Unlock()
		for _, dir := range dirs {
			if b, err := os.ReadFile(filepath.Join(dir, sid+".jsonl")); err == nil {
				return strings.ReplaceAll(string(b), `"sessionId":"`+sid+`"`, `"sessionId":"S"`)
			}
		}
		t.Fatalf("no session file %s", sid)
		return ""
	}
	conv := "[" + rmsg("user", "fix the bug")
	first := h.ask(conv + "]")
	lead := "lead" + pidOf(t, first) // the stand-in's session of a run started anew
	conv += "," + rmsg("assistant", first)
	prefixes := []string{conv}       // the conversation as far as each reply
	stood := []string{session(lead)} // the run's session after each reply
	for n := 2; n <= 4; n++ {
		conv += "," + rmsg("user", fmt.Sprintf("the conversation's turn %d", n))
		reply := h.ask(conv + "]")
		if pidOf(t, reply) != pidOf(t, first) {
			t.Fatalf("turn %d went to another run: %q", n, reply)
		}
		conv += "," + rmsg("assistant", reply)
		prefixes = append(prefixes, conv)
		stood = append(stood, session(lead))
	}
	for i := range 2 {
		task := fmt.Sprintf("the fork's task at reply %d", i+1)
		fork := h.ask(prefixes[i] + "," + rmsg("user", task) + "]")
		sid := ""
		for _, told := range h.toldBy() {
			if strings.Contains(told, "--resume ") && strings.Contains(told, task) {
				sid = strings.Fields(strings.SplitN(told, "--resume ", 2)[1])[0]
			}
		}
		if sid == "" || pidOf(t, fork) == pidOf(t, first) {
			t.Fatalf("the fork at reply %d went on from no copy of its own: %q", i+1, fork)
		}
		copied, want := session(sid), stood[i]
		if !strings.HasPrefix(copied, want) || strings.Count(copied, "\n") != strings.Count(want, "\n")+2 {
			t.Errorf("the fork at reply %d: want the run's session as it stood after that reply, then the fork's own turn\nstood:\n%s\ncopy:\n%s", i+1, want, copied)
		}
	}
	fifth := h.ask(conv + "," + rmsg("user", "the conversation's turn 5") + "]")
	if pidOf(t, fifth) != pidOf(t, first) || !strings.Contains(fifth, "turn 5") {
		t.Errorf("the run the forks branched off didn't go on with its next turn: %q", fifth)
	}
}

// A turn asked under another system prompt is no turn the run's session
// can go on with, and is told whole: a -p run's first message carries the
// client's system prompt, and a branch's run is told only the turn, so a
// fork under new instructions went on under the old ones and never heard
// the new. resume leaves such a turn too (turnKey). A review found it.
func TestClaudeTurnUnderOtherInstructionsIsNotBranched(t *testing.T) {
	h := newBusyRetryHarness(t, "print")
	t.Cleanup(h.s.subscription.abortAll)
	if err := os.WriteFile(filepath.Join(filepath.Dir(h.told), "nohold"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	task := rmsg("user", "fix the bug")
	first := h.ask(`[` + task + `]`)
	before := `[` + task + `,` + rmsg("assistant", first)
	second := h.ask(before + `,` + rmsg("user", "run the tests") + `]`)
	third := h.ask(before + `,` + rmsg("user", "run the tests") + `,` + rmsg("assistant", second) + `,` + rmsg("user", "commit it") + `]`)
	if pidOf(t, first) != pidOf(t, third) {
		t.Fatalf("the conversation's turns went to more than one run: %q, %q", first, third)
	}
	// the instructions are in the run's first message, not a file of its own
	h.s.subscription.mu.Lock()
	for _, run := range h.s.subscription.runs {
		if run.lean || run.appended || run.tui {
			h.s.subscription.mu.Unlock()
			t.Fatal("the run doesn't carry the client's instructions in its first message")
		}
	}
	h.s.subscription.mu.Unlock()

	const system = "ALWAYSANSWERINFRENCH"
	body := strings.Replace(h.body(before+`,`+rmsg("user", "run only the unit tests")+`]`), `"system":"be brief"`, `"system":"`+system+`"`, 1)
	rec := httptest.NewRecorder()
	var u Usage
	if code, why := h.s.serveClaudeSubscription(rec, httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body)), provider.Anthropic, h.p, "claude-sonnet-5", []byte(body), &u); code != 200 {
		t.Fatalf("%d %s", code, why)
	}
	var reply struct {
		Content []struct{ Text string } `json:"content"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &reply); err != nil || len(reply.Content) == 0 {
		t.Fatalf("no answer: %s", rec.Body)
	}
	told := h.toldBy()[pidOf(t, reply.Content[0].Text)]
	if !strings.Contains(told, system) {
		t.Fatalf("the turn's Claude Code was never told the new instructions:\n%.800s", told)
	}
}
