# 【翻译】使用 PLANS.md 解决多小时复杂问题

> 来源：OpenAI Cookbook，原文链接：https://github.com/openai/openai-cookbook/blob/main/articles/codex_exec_plans.md
> 说明：本文为中文翻译，保留原文中的代码、命令、配置片段和 mdnice 图片链接。

Codex 和 `gpt-5.2-codex` 模型（推荐）可以用来实现复杂任务，这类任务往往需要投入相当多的时间进行调研、设计与实现。本文介绍的方法，是一种提示模型去实现这类任务、并引导它成功完成整个项目的方式。

这些 plan 是详尽的设计文档，也是“活文档”。作为 Codex 的用户，你可以在 Codex 开始长时间实现之前，通过这些文档先验证它将采取的方案。下面给出的这个 `PLANS.md`，与一个真实案例中的文档非常相似；在那个案例里，Codex 仅凭一条 prompt 就持续工作了七个多小时。

要让 Codex 使用这些文档，我们首先会更新 `AGENTS.md`，说明何时使用 `PLANS.md`；然后当然还要把 `PLANS.md` 文件加入到仓库中。

## `AGENTS.md`

[`AGENTS.md`](https://github.com/openai/agents.md) 是一种简单的格式，用来指导像 Codex 这样的 coding agent。我们会定义一个术语，供用户作为简写使用，并给出一条关于何时使用 planning document 的简单规则。这里，我们把它称为 “ExecPlan”。注意，这只是一个任意命名的术语，Codex 并没有针对它进行过专门训练。有了这个简写之后，用户在 prompt 里就可以用它来把 Codex 引导到某个特定的 plan 定义上。

下面是 `AGENTS.md` 中一段用于指导 agent 何时使用 plan 的内容：

```md
# ExecPlans

When writing complex features or significant refactors, use an ExecPlan (as described in .agent/PLANS.md) from design to implementation.
```

## `PLANS.md`

下面是完整文档。这份文档中的提示措辞经过了精心设计，目的是向用户提供大量反馈，并引导模型精确地按照 plan 的要求完成实现。用户也可能会发现，根据自己的需要对这个文件进行定制、增删必要章节，会更有帮助。

~~~md
# Codex Execution Plans (ExecPlans):

本文档描述 execution plan（简称 “ExecPlan”）需要满足的要求。ExecPlan 是一种设计文档，coding agent 可以按照它来交付一个可工作的 feature 或系统改动。请把读者视为这个仓库的完全新手：他们手头只有当前工作树，以及你提供的这一个 ExecPlan 文件。不存在对以往 plan 的记忆，也没有任何外部上下文。

## How to use ExecPlans and PLANS.md

在编写可执行规范（ExecPlan）时，必须 _逐字遵循_ PLANS.md。如果它不在你的上下文中，就重新完整阅读整个 PLANS.md 文件，刷新记忆。为了产出准确的规范，必须彻底阅读（并反复重读）源材料。在创建规范时，应从骨架开始，并随着调研逐步充实内容。

在实现可执行规范（ExecPlan）时，不要向用户询问 “next steps”；只需直接推进到下一个 milestone。要让所有章节始终保持最新，在每一个暂停点都要补充或拆分列表项，明确说明已经取得的进展以及下一步要做什么。对歧义要自主作出判断，并且频繁 commit。

在讨论可执行规范（ExecPlan）时，要把所有决策记录到规范中的日志里，以便后续追溯；任何对规范的修改原因都必须清晰明确。ExecPlan 是活文档，而且应该始终能够做到：仅凭 ExecPlan 本身、而不依赖其他任何工作，就可以重新启动整个过程。

当为一个要求苛刻或存在大量未知数的设计做调研时，可以使用 milestone 来实现 proof of concept、“toy implementation”等内容，以验证用户提出的方案是否可行。你应当通过查找或获取依赖库来阅读它们的源代码，做深入研究，并把 prototype 纳入文档，以指导后续更完整的实现。

## Requirements

不可协商的要求：

* 每个 ExecPlan 都必须完全自包含。所谓自包含，是指它在当前形态下已经包含了新手成功完成任务所需的全部知识与指令。
* 每个 ExecPlan 都是活文档。随着进展推进、发现出现以及设计决策最终确定，贡献者必须持续修订它。每次修订后，它仍然必须保持完全自包含。
* 每个 ExecPlan 都必须让一个对本仓库毫无先验知识的完全新手，能够端到端实现这个 feature。
* 每个 ExecPlan 都必须产出可被证明正常工作的行为，而不仅仅是为了“满足某个定义”而修改代码。
* 每个 ExecPlan 都必须用平实语言定义每一个术语；如果做不到，就不要使用这个术语。

目的和意图优先。首先用几句话解释这项工作为什么重要，要从用户视角出发：这个改动完成后，用户能够做到什么以前做不到的事，以及如何观察它确实在工作。然后引导读者通过精确步骤达成这个结果，包括要修改什么、运行什么，以及他们应该看到什么现象。

执行你的 plan 的 agent 可以列出文件、读取文件、搜索、运行项目和执行测试。它不知道任何先前上下文，也不能从更早的 milestone 中推断你的本意。凡是你依赖的假设，都要重新写出来。不要把读者引向外部博客或文档；如果某些知识是必需的，就必须用你自己的话把它直接写进 plan 里。如果某个 ExecPlan 建立在先前的 ExecPlan 之上，并且那个文件已经提交入库，那么可以通过引用把它纳入；如果没有提交入库，你就必须把那个 plan 中所有相关上下文完整写进来。

## Formatting

格式与封装规则既简单又严格。每个 ExecPlan 都必须是一个带有 `md` 标记的单一 fenced code block，并且以三反引号开始、以三反引号结束。不要在其中再嵌套额外的三反引号代码块；如果需要展示命令、transcript、diff 或代码，应当把它们写成这个单一 code fence 内部的缩进块。为了避免意外提前结束 ExecPlan 的 code fence，请使用缩进而不是在 ExecPlan 内部继续嵌套代码 fence。

每个标题后要留两个换行，使用 #、## 等标准标题语法，并为有序列表和无序列表使用正确语法。

当你把 ExecPlan 写入某个 Markdown（`.md`）文件，并且这个文件的内容 *只有* 这一份 ExecPlan 时，应省略外层三反引号。

使用自然 prose。优先写句子，而不是堆砌列表。除非过于简短会导致意思不清，否则应避免 checklist、table 和冗长枚举。只有在 `Progress` 章节中允许使用 checklist，而且在那里它是强制要求。叙述性章节必须以 prose 为主。

## Guidelines

自包含和通俗语言是最重要的。如果你引入了一个不属于普通英语的短语（例如 “daemon”“middleware”“RPC gateway”“filter graph”），就要立刻给出定义，并提醒读者它在本仓库里是如何体现的（比如指出它出现在哪些文件或命令中）。不要写 “as defined previously” 或 “according to the architecture doc.”。即使需要重复，也要在这里直接给出必要解释。

避免常见失败模式。不要依赖未定义的术语。不要把 “feature 的字面要求” 描述得过于狭窄，以至于最后代码虽然能编译，却没有任何有意义的行为。不要把关键决策外包给读者。当存在歧义时，应当直接在 plan 中解决它，并解释为什么选择这条路径。宁可对用户可见效果解释得更充分，也不要把无关紧要的实现细节规定得过死。

用可观察结果来锚定 plan。说明实现完成后用户可以做什么、应运行哪些命令、应该看到什么输出。验收标准应当表述为人类可验证的行为（例如 “启动 server 后，访问 [http://localhost:8080/health](http://localhost:8080/health) 会返回 HTTP 200，body 为 OK”），而不是内部属性（例如 “added a HealthCheck struct”）。如果改动是内部性的，也要解释如何证明它的影响依然存在（例如运行某些测试，这些测试在改动前失败、改动后通过，并展示一个使用了新行为的场景）。

明确写出仓库上下文。使用完整的仓库相对路径来指明文件，精确写出函数名和模块名，并描述新文件应创建在什么位置。如果涉及多个区域，请加入一小段导览性说明，解释这些部分如何协同工作，从而让新手能够有把握地在仓库中导航。运行命令时，要写明 working directory 和完整命令行。如果结果依赖环境，就要说明你的假设，并在合理时给出替代方案。

保持幂等与安全。编写步骤时，应保证它们可以重复执行而不会造成破坏或漂移。如果某一步可能中途失败，应写明如何重试或调整。如果必须进行迁移或破坏性操作，就要明确说明备份方式或安全回退方案。优先选择可增量添加、可测试、可逐步验证的改动。

验证不是可选项。必须包含运行测试的说明；如适用，也要包含启动系统并观察其做出有用行为的说明。对于任何新 feature 或能力，都要描述全面的测试方式。给出预期输出和错误信息，使新手能够判断成功还是失败。只要可能，就应展示如何证明这个改动的效果不止是“能编译”（例如通过一个小型 end-to-end 场景、一次 CLI 调用，或一段 HTTP 请求/响应 transcript）。写明适用于该项目工具链的精确测试命令，以及如何解读其结果。

记录证据。当你的步骤会产生终端输出、简短 diff 或日志时，应把它们作为简洁示例包含在这个单一 fenced block 内。保持内容简短，只保留足以证明成功的部分。如果需要包含 patch，优先给出以文件为粒度的 diff 或简短摘录，让读者能够通过遵循你的说明自行复现，而不是直接粘贴大段内容。

## Milestones

Milestone 是叙事工具，而不是官僚流程。如果你把工作拆成多个 milestone，就要用一个简短段落引入每个 milestone，说明其范围、在该 milestone 结束时会新增什么、应运行哪些命令，以及你预期会观察到什么验收结果。要把它写成一个可读的故事：goal、work、result、proof。`Progress` 和 milestone 是不同的：milestone 负责讲述整体故事，`Progress` 负责跟踪细粒度工作。这两者都必须存在。不要仅仅为了简短而压缩某个 milestone，也不要省略那些可能对未来实现至关重要的细节。

每个 milestone 都必须能够独立验证，并以增量方式实现 execution plan 的整体目标。

## Living plans and design decisions

* ExecPlan 是活文档。每当你做出关键设计决策时，都要更新 plan，记录这个决策本身以及背后的思考。所有决策都必须记录在 `Decision Log` 章节中。
* ExecPlan 必须包含并持续维护 `Progress`、`Surprises & Discoveries`、`Decision Log` 和 `Outcomes & Retrospective` 章节。这些不是可选项。
* 当你发现了影响实现路径的 optimizer 行为、性能权衡、意外 bug，或 inverse/unapply 语义时，要把这些观察写入 `Surprises & Discoveries` 章节，并附上简短证据片段（理想情况下是测试输出）。
* 如果你在实现过程中途改变了方向，要在 `Decision Log` 中说明原因，并在 `Progress` 中反映其影响。Plan 既是你自己的 checklist，也是下一位贡献者的指南。
* 在完成一个重要任务或整个 plan 时，要写一条 `Outcomes & Retrospective` 记录，总结实现了什么、还剩下什么，以及从中学到了什么。

# Prototyping milestones and parallel implementations

当 prototype milestone 能为较大的改动降风险时，显式加入这类 milestone 是可以接受的，而且通常值得鼓励。例子包括：向某个依赖中添加一个底层 operator 来验证可行性，或者探索两种不同的组合顺序并测量 optimizer 效果。prototype 应保持为可增量添加、可测试的形式。要明确标注其范围是 “prototyping”；说明如何运行、如何观察结果；并写清楚将 prototype 提升为正式实现或将其丢弃的判定标准。

优先采用先增加、后删除、且始终保持测试通过的代码改动方式。并行实现（例如在迁移期间保留一个 adapter，与旧路径并存）是可以接受的，只要它们能降低风险，或使测试在大规模迁移期间继续通过。要说明如何验证两条路径，以及如何借助测试安全地退役其中一条。在同时引入多个新库或 feature area 时，可以考虑先创建若干 spike，分别独立评估这些能力的可行性，验证外部库是否符合预期，且能以隔离方式实现我们所需的功能。

## Skeleton of a Good ExecPlan

    # <简短、面向行动的描述>

    This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

    If PLANS.md file is checked into the repo, reference the path to that file here from the repository root and note that this document must be maintained in accordance with PLANS.md.

    ## Purpose / Big Picture

    用几句话解释这个改动完成后，别人能获得什么，以及他们如何确认它确实在工作。说明你将启用的用户可见行为。

    ## Progress

    使用带复选框的列表来概括细粒度步骤。每一个暂停点都必须记录在这里，即使这意味着要把一个部分完成的任务拆分成两个条目（“done” 与 “remaining”）。这一节必须始终反映工作的真实当前状态。

    - [x] (2025-10-01 13:00Z) Example completed step.
    - [ ] Example incomplete step.
    - [ ] Example partially completed step (completed: X; remaining: Y).

    使用时间戳来衡量进展速度。

    ## Surprises & Discoveries

    记录在实现过程中发现的意外行为、bug、优化点或洞见。提供简洁证据。

    - Observation: …
      Evidence: …

    ## Decision Log

    按照以下格式记录实现过程中做出的每一个决策：

    - Decision: …
      Rationale: …
      Date/Author: …

    ## Outcomes & Retrospective

    在重要 milestone 或全部完成时，总结结果、缺口和经验教训。把结果与最初的目的进行对照。

    ## Context and Orientation

    像面对一个对当前任务一无所知的读者那样，描述与本任务相关的当前状态。使用完整路径指出关键文件和模块。定义你将使用的任何不明显的术语。不要引用先前的 plan。

    ## Plan of Work

    用 prose 描述编辑与新增内容的顺序。对于每一项修改，都要写明文件和具体位置（函数、模块），以及需要插入或变更的内容。保持具体而简洁。

    ## Concrete Steps

    说明要运行的精确命令，以及应当在什么位置运行（working directory）。如果命令会产生输出，就展示一小段预期 transcript，方便读者比对。随着工作推进，这一节必须同步更新。

    ## Validation and Acceptance

    描述如何启动系统或触发相关行为，以及应该观察到什么。把验收写成行为描述，并包含明确输入与输出。如果涉及测试，就写成 “run <project’s test command> and expect <N> passed; the new test <name> fails before the change and passes after>”。

    ## Idempotence and Recovery

    如果步骤可以安全重复执行，就明确说明。如果某一步存在风险，就提供安全的重试或回滚路径。完成后保持环境整洁。

    ## Artifacts and Notes

    将最重要的 transcript、diff 或摘录以缩进示例的形式放在这里。保持简短，并聚焦于能证明成功的内容。

    ## Interfaces and Dependencies

    这里要有明确指令性。写清楚应使用哪些 library、module 和 service，以及原因。明确说明在该 milestone 结束时必须存在的 type、trait/interface 和函数签名。优先使用稳定的名称和路径，例如 `crate::module::function` 或 `package.submodule.Interface`。例如：

    In crates/foo/planner.rs, define:

        pub trait Planner {
            fn plan(&self, observed: &Observed) -> Vec<Action>;
        }

如果你遵循了以上指导，那么无论是一个单次、无状态的 agent，还是一个人类新手，都能够自上而下阅读你的 ExecPlan，并交付出一个可工作、可观察的结果。这才是标准：SELF-CONTAINED、SELF-SUFFICIENT、NOVICE-GUIDING、OUTCOME-FOCUSED。

当你修订一份 plan 时，必须确保你的修改在所有章节中都得到全面反映，包括那些“活文档”章节；并且你必须在 plan 末尾写一条说明，描述这次修改了什么，以及为什么修改。ExecPlan 不仅要说明做什么，也要尽可能说明为什么这样做。
~~~
