package gateway

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// turnBranchFixture is a bridge with one interactive run of own whose
// session on disk holds two turns, and which began its second turn after
// the first reply (line-1, m1): the conversation before that turn is before.
func turnBranchFixture(t *testing.T) (b *subscriptionBridge, run *subscriptionRun, own string, before []Message) {
	t.Helper()
	dir := t.TempDir()
	const sid = "lead-session"
	lines := []string{
		`{"type":"user","uuid":"ask-1","sessionId":"` + sid + `"}`,
		`{"type":"assistant","uuid":"line-1","sessionId":"` + sid + `","message":{"id":"m1","model":"claude-sonnet-5"}}`,
		`{"type":"user","uuid":"ask-2","sessionId":"` + sid + `"}`,
		`{"type":"assistant","uuid":"line-2","sessionId":"` + sid + `","message":{"id":"m2","model":"claude-sonnet-5"}}`,
	}
	if err := os.WriteFile(filepath.Join(dir, sid+".jsonl"), []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	before = []Message{tbUser("fix the bug"), tbSaid("fixed")}
	own = "claude\x00u"
	b = newSubscriptionBridge()
	run = &subscriptionRun{sessionID: sid, sessions: []string{dir}, tui: true, started: time.Now()}
	run.owner.Store(&own)
	// the key resume found the run waiting at, for a request of these settings
	run.turns = []turnStart{{key: turnKey(own, tbRequest(before), before), line: "line-1", msg: "m1"}}
	b.runs["r"] = run
	return b, run, own, before
}

// tbRequest is a request of the fixture's settings with msgs.
func tbRequest(msgs []Message) *Request {
	return &Request{Model: "claude-sonnet-5", System: "be brief", Messages: msgs}
}

func tbUser(s string) Message { return Message{Role: "user", Parts: []Part{{Kind: Text, Text: s}}} }
func tbSaid(s string) Message {
	return Message{Role: "assistant", Parts: []Part{{Kind: Text, Text: s}}}
}

// cutAsFarAsTheFirstReply checks that s is a copy of the fixture's session
// of its own, cut after the first reply, told the turn alone.
func cutAsFarAsTheFirstReply(t *testing.T, run *subscriptionRun, s *savedSession) {
	t.Helper()
	if s.session == run.sessionID || len(s.since) != 1 || !s.tui {
		t.Fatalf("branch = session %q, %d messages told, tui %v; want a copy of its own, told the turn alone, interactive", s.session, len(s.since), s.tui)
	}
	copied, _ := os.ReadFile(filepath.Join(run.sessions[0], s.session+".jsonl"))
	if c := string(copied); !strings.Contains(c, "line-1") || strings.Contains(c, "ask-2") || strings.Contains(c, "line-2") {
		t.Fatalf("the copy keeps %q, want the session as far as the first reply", c)
	}
}

// turnBranch cuts the session of the run that began a turn where the
// request goes on from, after the reply it ended at, and takes none where
// that isn't the case: another account's run, a conversation no turn began
// at, the same conversation under another system prompt or model (resume
// wouldn't give the run that turn either), tool results for a call a reply
// made, a run taken back, or a -p run while the interactive mode is on.
func TestClaudeTurnBranchCutsAfterTheReplyTheTurnFollows(t *testing.T) {
	b, run, own, before := turnBranchFixture(t)
	fork := tbRequest(append(append([]Message{}, before...), tbUser("update the docs")))
	got := b.turnBranch(fork, own)
	if got == nil {
		t.Fatal("no branch for a turn after the reply a turn of the run began at")
	}
	t.Cleanup(got.discard)
	cutAsFarAsTheFirstReply(t, run, got)

	if got := b.turnBranch(fork, "claude\x00v"); got != nil {
		got.discard()
		t.Error("a run of another account was branched")
	}
	later := tbRequest(append(append([]Message{}, before...), tbUser("and run the tests"), tbSaid("they pass"), tbUser("commit")))
	if got := b.turnBranch(later, own); got != nil {
		got.discard()
		t.Error("a turn after a reply no turn began at was branched")
	}
	other := tbRequest(fork.Messages)
	other.System = "answer in French"
	if got := b.turnBranch(other, own); got != nil {
		got.discard()
		t.Error("a turn under another system prompt was branched off a session told the old one")
	}
	other = tbRequest(fork.Messages)
	other.Model = "claude-opus-5"
	if got := b.turnBranch(other, own); got != nil {
		got.discard()
		t.Error("a turn asked of another model was branched")
	}
	// the results of a call its run's reply made go on with that run, or
	// branch at the call (branch): no new turn
	results := tbRequest([]Message{tbUser("fix the bug"),
		{Role: "assistant", Parts: []Part{{Kind: ToolCall, ID: "toolu_1", Name: "read", Args: []byte(`{}`)}}},
		{Role: "user", Parts: []Part{{Kind: ToolResult, CallID: "toolu_1", Text: "ok"}}}})
	if got := b.turnBranch(results, own); got != nil {
		got.discard()
		t.Error("tool results were branched as a new turn")
	}

	claudeTUIRuns = true
	t.Cleanup(func() { claudeTUIRuns = false })
	run.mu.Lock()
	run.tui = false
	run.mu.Unlock()
	if got := b.turnBranch(fork, own); got != nil {
		got.discard()
		t.Error("a -p run's session was branched while the interactive mode is on")
	}
	run.mu.Lock()
	run.tui, run.rewound = true, true
	run.mu.Unlock()
	if got := b.turnBranch(fork, own); got != nil {
		got.discard()
		t.Error("a run taken back was branched")
	}
}

// A run that ended between turns — its conversation told to a new run
// (retire), its reply ended by a refusal — has no call line and no turn
// under way, only the starts of the turns it answered: its session is kept
// for those (keepable), and a fork at one of them branches off it as off a
// live run's.
func TestClaudeTurnBranchGoesOnFromAnEndedRunsSession(t *testing.T) {
	b, run, own, before := turnBranchFixture(t)
	t.Cleanup(b.dropAllEnded)
	delete(b.runs, "r")
	run.mu.Lock()
	run.closed = true
	run.mu.Unlock()
	if !run.keepable() {
		t.Fatal("a run with only its turns' starts to cut at isn't kept")
	}
	b.mu.Lock()
	b.keepEnded(run)
	b.mu.Unlock()
	fork := tbRequest(append(append([]Message{}, before...), tbUser("update the docs")))
	got := b.turnBranch(fork, own)
	if got == nil {
		t.Fatal("no branch off the kept session of an ended run")
	}
	t.Cleanup(got.discard)
	cutAsFarAsTheFirstReply(t, run, got)

	// with no turn start either, there is nothing to cut at
	run.mu.Lock()
	run.turns = nil
	run.mu.Unlock()
	if run.keepable() {
		t.Error("a run with nothing to cut at is kept")
	}
}
