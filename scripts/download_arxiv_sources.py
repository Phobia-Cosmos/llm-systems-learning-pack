#!/usr/bin/env python3
"""Find arXiv records for local papers and download their TeX sources.

The local PDF naming convention is expected to look like:

    2022CVPR-Online Continual Learning on a Contaminated Data Stream.pdf

Each downloaded source is extracted into a directory using the same
``YEARVENUE-Official arXiv Title`` convention.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import gzip
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree


ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_EPRINT = "https://arxiv.org/e-print/{arxiv_id}"
USER_AGENT = "local-arxiv-source-downloader/1.0"
GENERIC_VENUES = {"arxiv", "report", "openai", "nvidia", "unknown"}


@dataclass(frozen=True)
class LocalPaper:
    pdf_path: str
    year: str
    venue: str
    title: str

    @property
    def prefix(self) -> str:
        return f"{self.year}{self.venue}" if self.year and self.venue else ""


@dataclass(frozen=True)
class ArxivMatch:
    arxiv_id: str
    title: str
    published_year: str
    score: float
    source: str


class ArxivClient:
    def __init__(self, delay: float = 3.1, retries: int = 4) -> None:
        self.delay = delay
        self.retries = retries
        self.last_request = 0.0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def open(self, url: str):
        for attempt in range(self.retries):
            self._wait()
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                response = urllib.request.urlopen(request, timeout=90)
                self.last_request = time.monotonic()
                return response
            except urllib.error.HTTPError as error:
                self.last_request = time.monotonic()
                if error.code not in {429, 500, 502, 503, 504} or attempt + 1 == self.retries:
                    raise
            except urllib.error.URLError:
                self.last_request = time.monotonic()
                if attempt + 1 == self.retries:
                    raise
            time.sleep(max(self.delay, 2**attempt))
        raise RuntimeError(f"request failed: {url}")

    def search(self, title: str, year: str, threshold: float) -> ArxivMatch | None:
        terms = " ".join(re.findall(r"[\w]+", title, flags=re.UNICODE))
        queries = [f'ti:"{terms}"', f'all:"{terms}"']
        candidates: dict[str, ArxivMatch] = {}

        for query in queries:
            params = urllib.parse.urlencode(
                {"search_query": query, "start": 0, "max_results": 10}
            )
            with self.open(f"{ARXIV_API}?{params}") as response:
                feed = ElementTree.fromstring(response.read())
            for entry in feed.findall(f"{ATOM}entry"):
                candidate = parse_arxiv_entry(entry, title, year)
                previous = candidates.get(candidate.arxiv_id)
                if previous is None or candidate.score > previous.score:
                    candidates[candidate.arxiv_id] = candidate
            if candidates and max(item.score for item in candidates.values()) >= threshold:
                break

        if not candidates:
            return None
        best = max(candidates.values(), key=lambda item: item.score)
        return best if best.score >= threshold else None

    def download(self, arxiv_id: str, destination: Path) -> None:
        url = ARXIV_EPRINT.format(arxiv_id=urllib.parse.quote(arxiv_id, safe="/"))
        curl = shutil.which("curl")
        if curl:
            self._wait()
            try:
                subprocess.run(
                    [
                        curl,
                        "-L",
                        "--fail",
                        "--silent",
                        "--show-error",
                        "--retry",
                        str(self.retries),
                        "--retry-delay",
                        str(max(1, round(self.delay))),
                        "--connect-timeout",
                        "20",
                        "--max-time",
                        "600",
                        "--user-agent",
                        USER_AGENT,
                        "--output",
                        str(destination),
                        url,
                    ],
                    check=True,
                )
            finally:
                self.last_request = time.monotonic()
            return
        with self.open(url) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("&", " and ")
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def title_score(expected: str, candidate: str) -> float:
    left = normalize_title(expected)
    right = normalize_title(candidate)
    if not left or not right:
        return 0.0
    sequence = difflib.SequenceMatcher(None, left, right).ratio()
    left_words = set(left.split())
    right_words = set(right.split())
    token_f1 = 2 * len(left_words & right_words) / (len(left_words) + len(right_words))
    score = 0.65 * sequence + 0.35 * token_f1
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) >= 24 and longer.startswith(shorter):
        score = max(score, 0.90)
    return score


def parse_arxiv_entry(entry: ElementTree.Element, expected_title: str, expected_year: str) -> ArxivMatch:
    entry_id = (entry.findtext(f"{ATOM}id") or "").strip()
    arxiv_id = re.sub(r"v\d+$", "", entry_id.rsplit("/", 1)[-1])
    title = re.sub(r"\s+", " ", entry.findtext(f"{ATOM}title") or "").strip()
    published = (entry.findtext(f"{ATOM}published") or "")[:4]
    score = title_score(expected_title, title)
    if expected_year.isdigit() and published.isdigit():
        difference = abs(int(expected_year) - int(published))
        if difference <= 1:
            score = min(1.0, score + 0.02)
        elif difference >= 4:
            score = max(0.0, score - 0.04)
    return ArxivMatch(arxiv_id, title, published, round(score, 4), "api-search")


def parse_local_paper(pdf_path: Path) -> LocalPaper:
    stem = pdf_path.stem.strip()
    match = re.match(r"^((?:19|20)\d{2})([^-]+)-(.+)$", stem)
    if match:
        return LocalPaper(
            str(pdf_path),
            match.group(1),
            match.group(2).strip(),
            match.group(3).strip().rstrip("."),
        )
    return LocalPaper(str(pdf_path), "", "", stem.rstrip("."))


def venue_rank(venue: str) -> int:
    return 0 if venue.casefold() in GENERIC_VENUES else 1


def discover_papers(papers_root: Path) -> tuple[list[LocalPaper], int]:
    selected: dict[str, LocalPaper] = {}
    duplicate_count = 0
    for pdf_path in sorted(papers_root.rglob("*.pdf")):
        paper = parse_local_paper(pdf_path)
        key = normalize_title(paper.title)
        current = selected.get(key)
        if current is None:
            selected[key] = paper
            continue
        duplicate_count += 1
        current_quality = (venue_rank(current.venue), bool(current.prefix), len(current.title))
        paper_quality = (venue_rank(paper.venue), bool(paper.prefix), len(paper.title))
        if paper_quality > current_quality:
            selected[key] = paper
    return sorted(selected.values(), key=lambda item: (item.year, item.venue, item.title)), duplicate_count


def extract_arxiv_id(url: str) -> str:
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", url, flags=re.IGNORECASE)
    if not match:
        return ""
    arxiv_id = match.group(1).removesuffix(".pdf").strip("/")
    return re.sub(r"v\d+$", "", arxiv_id)


def load_known_arxiv(index_path: Path) -> list[tuple[str, str, str]]:
    if not index_path.exists():
        return []
    entries: list[tuple[str, str, str]] = []
    with index_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            arxiv_id = extract_arxiv_id(row.get("url", ""))
            title = row.get("title", "").strip()
            if arxiv_id and title:
                entries.append((normalize_title(title), title, arxiv_id))
    return entries


def match_known_arxiv(
    paper: LocalPaper, entries: Iterable[tuple[str, str, str]], threshold: float = 0.92
) -> ArxivMatch | None:
    normalized = normalize_title(paper.title)
    exact = [entry for entry in entries if entry[0] == normalized]
    if exact:
        _, title, arxiv_id = exact[0]
        return ArxivMatch(arxiv_id, title, arxiv_year(arxiv_id), 1.0, "paper_list.tsv")

    best: tuple[float, str, str] | None = None
    for _, title, arxiv_id in entries:
        score = title_score(paper.title, title)
        if best is None or score > best[0]:
            best = (score, title, arxiv_id)
    if best and best[0] >= threshold:
        return ArxivMatch(best[2], best[1], arxiv_year(best[2]), round(best[0], 4), "paper_list.tsv")
    return None


def arxiv_year(arxiv_id: str) -> str:
    match = re.match(r"(\d{2})(?:\d{2})?\.", arxiv_id)
    if not match:
        return ""
    value = int(match.group(1))
    return str(1900 + value if value >= 91 else 2000 + value)


def safe_name(value: str, max_length: int = 180) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("/", "-").replace("\\", "-").replace("\x00", "")
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value[:max_length].rstrip(" .") or "untitled"


def output_name(paper: LocalPaper, match: ArxivMatch) -> str:
    year = paper.year or match.published_year or arxiv_year(match.arxiv_id)
    venue = paper.venue or "arXiv"
    return safe_name(f"{year}{venue}-{match.title}")


def safe_archive_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"unsafe archive path: {member_name}")
    return target


def extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as source:
        for member in source.getmembers():
            target = safe_archive_target(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = source.extractfile(member)
                if extracted is not None:
                    with extracted, target.open("wb") as output:
                        shutil.copyfileobj(extracted, output)


def extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = safe_archive_target(destination, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as input_file, target.open("wb") as output:
                    shutil.copyfileobj(input_file, output)


def flatten_single_directory(destination: Path) -> None:
    entries = list(destination.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        return
    nested = entries[0]
    for child in list(nested.iterdir()):
        shutil.move(str(child), destination / child.name)
    nested.rmdir()


def extract_source(archive: Path, destination: Path) -> str:
    if tarfile.is_tarfile(archive):
        extract_tar(archive, destination)
        archive_type = "tar"
    elif zipfile.is_zipfile(archive):
        extract_zip(archive, destination)
        archive_type = "zip"
    else:
        data = archive.read_bytes()
        if data.startswith(b"%PDF"):
            raise ValueError("arXiv returned a PDF instead of source files")
        if data.startswith(b"\x1f\x8b"):
            data = gzip.decompress(data)
            archive_type = "gzip-tex"
        else:
            archive_type = "plain-tex"
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("download is neither a supported archive nor TeX text") from error
        (destination / "main.tex").write_bytes(data)
    flatten_single_directory(destination)
    return archive_type


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def install_source(
    client: ArxivClient,
    paper: LocalPaper,
    match: ArxivMatch,
    output_root: Path,
    force: bool,
    keep_archive: bool,
) -> tuple[Path, str]:
    destination = output_root / output_name(paper, match)
    if destination.exists() and not force:
        return destination, "already-exists"

    staging = Path(tempfile.mkdtemp(prefix=".arxiv-source-", dir=output_root))
    archive = staging.parent / f"{staging.name}.download"
    try:
        client.download(match.arxiv_id, archive)
        archive_type = extract_source(archive, staging)
        metadata = {
            "local_paper": asdict(paper),
            "arxiv": asdict(match),
            "source_url": ARXIV_EPRINT.format(arxiv_id=match.arxiv_id),
            "archive_type": archive_type,
            "downloaded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        atomic_json(staging / "metadata.json", metadata)
        if keep_archive:
            shutil.move(archive, staging / "arxiv-source.download")
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
        return destination, "downloaded"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        archive.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(
        description="Search arXiv for PDFs under ~/Desktop/ai/papers and download TeX sources sequentially."
    )
    parser.add_argument("--papers-root", type=Path, default=home / "Desktop/ai/papers")
    parser.add_argument("--output", type=Path, default=home / "Downloads/arxiv_sources")
    parser.add_argument("--index", type=Path, default=home / "Desktop/ai/paper_list.tsv")
    parser.add_argument("--force-search", action="store_true", help="Ignore arXiv IDs already recorded in paper_list.tsv")
    parser.add_argument("--force", action="store_true", help="Replace existing source directories")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the raw e-print download after extraction")
    parser.add_argument("--search-only", action="store_true", help="Search and write the manifest without downloading sources")
    parser.add_argument("--dry-run", action="store_true", help="Only print extracted local paper names")
    parser.add_argument("--limit", type=int, help="Process at most this many unique papers")
    parser.add_argument("--match", help="Only process local titles containing this text")
    parser.add_argument("--delay", type=float, default=3.1, help="Minimum seconds between arXiv requests")
    parser.add_argument("--threshold", type=float, default=0.80, help="Minimum title similarity for API search results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    papers_root = args.papers_root.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not papers_root.is_dir():
        raise SystemExit(f"papers directory not found: {papers_root}")

    papers, duplicate_count = discover_papers(papers_root)
    if args.match:
        query = normalize_title(args.match)
        papers = [paper for paper in papers if query in normalize_title(paper.title)]
    if args.limit is not None:
        papers = papers[: max(0, args.limit)]

    print(f"Found {len(papers)} unique papers ({duplicate_count} duplicate PDF names removed).")
    if args.dry_run:
        for index, paper in enumerate(papers, 1):
            prefix = paper.prefix or "<year+venue from arXiv>"
            print(f"{index:03d}  {prefix}-{paper.title}")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    known_entries = [] if args.force_search else load_known_arxiv(args.index.expanduser())
    client = ArxivClient(delay=max(0.0, args.delay))
    results: list[dict] = []
    manifest_path = output_root / "manifest.json"

    for index, paper in enumerate(papers, 1):
        print(f"[{index}/{len(papers)}] {paper.prefix or '?'}-{paper.title}", flush=True)
        result = {"paper": asdict(paper), "status": "pending"}
        try:
            match = match_known_arxiv(paper, known_entries)
            if match is None:
                match = client.search(paper.title, paper.year, args.threshold)
            if match is None:
                result["status"] = "not-found"
                print("  not found with a sufficiently similar arXiv title", flush=True)
            else:
                result["arxiv"] = asdict(match)
                result["directory_name"] = output_name(paper, match)
                if args.search_only:
                    result["status"] = "matched"
                    print(f"  {match.arxiv_id}  score={match.score:.3f}  {match.title}", flush=True)
                else:
                    destination, status = install_source(
                        client, paper, match, output_root, args.force, args.keep_archive
                    )
                    result["status"] = status
                    result["output"] = str(destination)
                    print(f"  {status}: {destination.name}", flush=True)
        except (OSError, ValueError, urllib.error.URLError, subprocess.CalledProcessError) as error:
            result["status"] = "error"
            result["error"] = str(error)
            print(f"  error: {error}", flush=True)
        results.append(result)
        atomic_json(
            manifest_path,
            {
                "papers_root": str(papers_root),
                "output_root": str(output_root),
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "results": results,
            },
        )

    counts: dict[str, int] = {}
    for result in results:
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1
    print("Finished: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    print(f"Manifest: {manifest_path}")
    return 0 if not counts.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
