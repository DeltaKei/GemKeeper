# ==============================================================================
# GemKeeper v4.3 - Throttled Git Push
# Requires: pip install pyyaml playwright beautifulsoup4
# ==============================================================================

import asyncio
import os
import re
import json
import hashlib
import sys
from datetime import datetime, timedelta
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import yaml
import git # GitPython

# ==============================================================================
# Configuration Loader
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml") # 개인 설정
SELECTORS_PATH = os.path.join(SCRIPT_DIR, "selectors.yaml") # 공유 셀렉터

def load_config():
    config = {}
    
    # 1. Load Selectors (필수)
    if not os.path.exists(SELECTORS_PATH):
        print(f"[Critical Error] Selectors file not found at: {SELECTORS_PATH}")
        sys.exit(1)
    try:
        with open(SELECTORS_PATH, 'r', encoding='utf-8') as f:
            sel_data = yaml.safe_load(f)
            config.update(sel_data) # 'selectors' 키가 들어감
    except Exception as e:
        print(f"[Error] Failed to load selectors.yaml: {e}")
        sys.exit(1)

    # 2. Load User Config (필수)
    if not os.path.exists(CONFIG_PATH):
        print(f"[Critical Error] User config file not found at: {CONFIG_PATH}")
        print(f"Please copy 'config_template.yaml' to 'config.yaml' and set your preferences.")
        sys.exit(1)
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            sys_data = yaml.safe_load(f)
            config.update(sys_data) # 'system' 키가 들어감
    except Exception as e:
        print(f"[Error] Failed to load config.yaml: {e}")
        sys.exit(1)

    return config

CONFIG = load_config()

# Extract System Constants
CDP_URL = CONFIG['system']['cdp_url']
KB_ROOT = os.path.join(SCRIPT_DIR, CONFIG['system']['kb_folder_name'])
TARGET_DOMAIN = CONFIG['system']['target_domain']
POLL_INTERVAL = CONFIG['system']['poll_interval']
DOM_Watchdog_Threshold = CONFIG['system'].get('dom_watchdog_threshold', 5)

# ==============================================================================
# Utility Functions
# ==============================================================================

def sanitize_filename(name):
    """
    Sanitizes filenames to be safe for file systems.
    """
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', name)
    safe_name = re.sub(r'\s+', ' ', safe_name)
    safe_name = re.sub(r'_+', '_', safe_name)
    return safe_name.strip('_ ')[:100]

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def cleanup_stale_git_locks(repo_path):
    """
    Removes stale Git lock files that might prevent operations.
    Run this at startup to clear locks left by previous crashed runs.
    """
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.isdir(git_dir):
        return

    lock_files = [
        "index.lock",
        "HEAD.lock",
        "refs/heads/master.lock",
        "refs/heads/main.lock",
        "config.lock"
    ]

    for lock_file in lock_files:
        lock_path = os.path.join(git_dir, lock_file)
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
                print(f"[Git] Removed stale lock file: {lock_file}")
            except Exception as e:
                print(f"[Git] Failed to remove lock file {lock_file}: {e}")

def generate_message_id(role, text_content):
    """
    Generates a deterministic SHA256 hash based on role and text content.
    """
    normalized_text = text_content.strip()
    raw_key = f"{role}:{normalized_text}".encode('utf-8')
    return hashlib.sha256(raw_key).hexdigest()

def clean_html(element):
    """
    Deep Cleaning: Removes UI noise using selectors from CONFIG.
    """
    selectors = CONFIG['selectors']
    
    # Remove by Class
    noise_classes = selectors.get('noise_classes', [])
    for cls in noise_classes:
        for tag in element.find_all(class_=re.compile(cls)):
            tag.decompose()

    # Remove by Tag
    noise_tags = selectors.get('noise_tags', [])
    for tag_name in noise_tags:
        for tag in element.find_all(tag_name):
            tag.decompose()

    # Remove tooltips
    tooltip_pattern = selectors.get('tooltip_pattern', 'tooltip')
    for tag in element.find_all(class_=re.compile(tooltip_pattern)):
        tag.decompose()

    return element

def extract_chat_id(url):
    """
    Extracts the Chat ID from the URL.
    Format: https://gemini.google.com/app/abcd12345
    """
    match = re.search(r"/app/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    return None

# ==============================================================================
# Markdown Converter
# ==============================================================================

def traverse_dom(element, indent_level=0):
    """
    Recursive DOM traverser that converts HTML elements to Markdown.
    """
    md_output = ""
    
    if element.name is None:
        text = element.string
        if not text: return ""
        text = text.strip()
        if not text: return ""
        return text

    tag = element.name.lower()
    
    if tag in ['p', 'div', 'section', 'article']:
        for child in element.children:
            child_text = traverse_dom(child, indent_level)
            if child_text:
                if md_output and not md_output.endswith(('\n', ' ')):
                    md_output += " "
                md_output += child_text
        md_output = md_output.strip() + "\n\n"
        
    elif tag == 'br':
        md_output += "\n"
        
    elif tag in ['ul', 'ol']:
        for i, child in enumerate(element.find_all('li', recursive=False)):
            prefix = "* " if tag == 'ul' else f"{i+1}. "
            content = traverse_dom(child, indent_level + 1).strip()
            md_output += f"{'  ' * indent_level}{prefix}{content}\n"
        md_output += "\n"
        
    elif tag == 'li':
        for child in element.children:
            md_output += traverse_dom(child, indent_level) + " "
            
    elif tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        level = int(tag[1])
        content = ""
        for child in element.children:
            content += traverse_dom(child).strip() + " "
        md_output += f"{'#' * level} {content.strip()}\n\n"
        
    elif tag == 'pre':
        code_content = element.get_text()
        lang = ""
        if element.has_attr('class'):
            for c in element['class']:
                if 'language-' in c:
                    lang = c.replace('language-', '')
        md_output += f"\n```{lang}\n{code_content.strip()}\n```\n\n"
        
    elif tag == 'code':
        if element.parent.name != 'pre':
            md_output += f"`{element.get_text().strip()}`"
            
    elif tag in ['b', 'strong']:
        content = ""
        for child in element.children:
            content += traverse_dom(child)
        md_output += f"**{content.strip()}**"
        
    elif tag in ['i', 'em']:
        content = ""
        for child in element.children:
            content += traverse_dom(child)
        md_output += f"*{content.strip()}*"
        
    elif tag == 'a':
        text = ""
        for child in element.children:
            text += traverse_dom(child)
        href = element.get('href', '')
        md_output += f"[{text.strip()}]({href})"
        
    elif tag == 'img':
        src = element.get('src')
        alt = element.get('alt', 'Image')
        if src and src.startswith('http'):
            md_output += f"![{alt}]({src})\n"

    else:
        content = ""
        for child in element.children:
            child_text = traverse_dom(child, indent_level)
            if child_text:
                if content and not content.endswith(('\n', ' ')):
                    content += " "
                content += child_text
        md_output += content

    return md_output

def generate_markdown_file(title, history, date_str):
    lines = []
    lines.append(f"# {title}")
    lines.append(f"**Last Sync:** {date_str}\n")
    lines.append(f"**V4.3 Session Log** (Watchdog + Sidebar Title + Throttled Push)\n")

    for msg in history:
        role_header = "## 👤 User" if msg['role'] == 'user' else "## 🤖 Gemini"
        
        if msg.get('html'):
            soup = BeautifulSoup(msg['html'], 'html.parser')
            body = traverse_dom(soup)
        else:
            body = msg.get('text', '')

        lines.append(f"{role_header}\n{body}\n")

    return "\n".join(lines)


# ==============================================================================
# Core Scraping & State Logic
# ==============================================================================

async def scrape_page(page, smart_title_fallback=False, current_chat_id=None):
    """
    Extracts messages using selectors from CONFIG.
    Also parses Sidebar for accurate Title.
    """
    try:
        content = await page.content()
    except:
        return [] # Page might be closed

    soup = BeautifulSoup(content, 'html.parser')
    
    messages = []
    candidate_title = None
    sidebar_title = None
    header_title = None

    # --- 1. Message Scraping ---
    msg_class = CONFIG['selectors'].get('message_content_class', 'message-content')
    all_elements = soup.find_all(['div', msg_class])
    
    user_ids = CONFIG['selectors']['user_identifiers']
    model_ids = CONFIG['selectors']['model_identifiers']
    stable_id_pat = CONFIG['selectors']['stable_id_pattern']

    for el in all_elements:
        if el.parent is None: continue
        if not hasattr(el, 'attrs') or el.attrs is None: continue

        classes = el.get('class', [])
        if not classes: continue
        class_str = " ".join(classes)
        
        role = None
        
        if any(uid in class_str for uid in user_ids):
            role = 'user'
        elif any(mid in class_str for mid in model_ids):
            role = 'model'
            
        if role:
            stable_id = None
            if el.has_attr('id'):
                raw_id = el['id']
                if stable_id_pat in raw_id:
                    stable_id = raw_id
            
            clean_html(el)
            
            if role == 'user':
                q_class = CONFIG['selectors'].get('query_text_class', 'query-text')
                text_el = el.find(class_=q_class) or el
                text_content = text_el.get_text(strip=True)
                html_content = str(el) 
                
                if smart_title_fallback and not candidate_title:
                    candidate_title = text_content[:30].replace("\n", " ").strip()
            else: # model
                html_content = str(el)
                text_content = el.get_text(strip=True)

            if text_content:
                final_id = stable_id if stable_id else generate_message_id(role, text_content)
                msg_obj = {
                    "id": final_id,
                    "role": role,
                    "text": text_content,
                    "html": html_content,
                    "timestamp": datetime.now().isoformat()
                }
                messages.append(msg_obj)
    
    # --- 2. Header Title Extraction (Final V4.2) ---
    try:
        # Selector: conversation-actions .conversation-title
        header_sel = CONFIG['selectors'].get('header_title', 'conversation-actions .conversation-title')
        header_el = soup.select_one(header_sel)
        if header_el:
            t = header_el.get_text(strip=True)
            if t:
                header_title = t
    except Exception as e:
         pass

    if smart_title_fallback:
        return messages, candidate_title, header_title
    return messages


class GemSessionV4:
    def __init__(self, session_id):
        self.session_id = sanitize_filename(session_id)
        
        self.session_dir = os.path.join(KB_ROOT, "Sessions", self.session_id)
        ensure_directory(self.session_dir)
        
        self.json_path = os.path.join(self.session_dir, "history.json")
        # Default filename. Will be updated if title is set.
        self.md_path = os.path.join(self.session_dir, "session_full.md") 
        
        self.history = self._load_history()
        # Fix: ensure history is always a list
        if not isinstance(self.history, list):
             self.history = []
        self.message_map = {msg['id']: i for i, msg in enumerate(self.history) if isinstance(msg, dict) and 'id' in msg}
        
        self.current_title = f"Session_{self.session_id}" 
        
        # Try to recover title from existing MD files if any
        self._recover_title_from_files()

        # Git Integration (Init)
        self.GIT_REPO_URL = os.environ.get("GIT_REPO_URL")
        # Ensure we treat KB_ROOT as the git repo, not just the session dir
        self.repo_root = KB_ROOT 
        
        if self.GIT_REPO_URL:
            self.initialize_git()
        else:
            print("[Warning] GIT_REPO_URL not set. Automated Git backup disabled.")

        print(f"\n[Session] Active: {self.session_id}")
        print(f"          Path: {self.session_dir}")
        print(f"          Loaded: {len(self.history)} existing messages")

    def initialize_git(self):
        """Initializes Git repo at KB_ROOT and checks remote."""
        if not self.GIT_REPO_URL:
            return

        try:
            try:
                repo = git.Repo(self.repo_root)
            except git.exc.InvalidGitRepositoryError:
                # Init new if not exists
                repo = git.Repo.init(self.repo_root)
                print(f"[Git] Initialized new repository at {self.repo_root}")

            # Check/Add Remote
            if 'origin' not in [remote.name for remote in repo.remotes]:
                repo.create_remote('origin', self.GIT_REPO_URL)
                print(f"[Git] Remote 'origin' added: {self.GIT_REPO_URL}")
            
            # Try pull to sync (prevent conflicts)
            try:
                repo.git.pull('origin', 'main', allow_unrelated_histories=True)
            except:
                # Fallback or empty repo
                pass
                
        except Exception as e:
            print(f"[Git Init Error] {e}")

    def commit_local(self, message):
        """Commits changes locally without pushing."""
        if not self.GIT_REPO_URL:
            return
            
        try:
            repo = git.Repo(self.repo_root)
            
            # Add all changes in KB_ROOT
            repo.git.add('.')
            
            # Check for changes
            if not repo.is_dirty(untracked_files=True):
                # print("[Git] No changes to commit.")
                return

            # Commit
            repo.index.commit(message)
            print(f"[Git] Committed (Local): {message}")
            
        except Exception as e:
            print(f"[Git Commit Error] {e}")

    def push_remote(self):
        """Executes git push to remote."""
        if not self.GIT_REPO_URL:
            return False
            
        try:
            repo = git.Repo(self.repo_root)
            
            # Push (Main or Master)
            try:
                repo.git.push('origin', 'main')
            except:
                repo.git.push('origin', 'master')
            
            print(f"[Git] Pushed to GitHub at {datetime.now().strftime('%H:%M:%S')}")
            return True

        except Exception as e:
            print(f"[Git Push Error] {e}")
            return False

    def _recover_title_from_files(self):
        # Look for any .md file that isn't session_full.md
        try:
            files = [f for f in os.listdir(self.session_dir) if f.endswith('.md')]
            for f in files:
                if f != "session_full.md":
                    # Assume this is the title
                    name_part = f[:-3] # remove .md
                    self.current_title = name_part
                    self.md_path = os.path.join(self.session_dir, f)
                    return
        except:
            pass

    def _load_history(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    return []
            except Exception as e:
                print(f"[Error] Failed to load JSON: {e}")
                return []
        return []

    def set_title(self, title):
        if not title: return
        safe_title = sanitize_filename(title)
        
        # If title is effectively different
        if safe_title != self.current_title and safe_title != f"Session_{self.session_id}":
            old_md_path = self.md_path
            new_md_path = os.path.join(self.session_dir, f"{safe_title}.md")
            
            # 1. Update Internal State
            self.current_title = safe_title
            self.md_path = new_md_path
            
            # 2. Rename File on Disk if it exists
            if os.path.exists(old_md_path) and not os.path.exists(new_md_path):
                try:
                    os.rename(old_md_path, new_md_path)
                    print(f"[Rename] {os.path.basename(old_md_path)} -> {os.path.basename(new_md_path)}")
                except Exception as e:
                    print(f"[Rename Error] Could not rename MD file: {e}")
            elif os.path.exists(new_md_path) and old_md_path != new_md_path:
                 # Logic to handle if target already exists (maybe merge? or just switch usage)
                 # For now, just switch usage to the new file
                 pass

    def save(self):
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            
            md_content = generate_markdown_file(
                self.current_title, 
                self.history, 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            with open(self.md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            # Git Trigger: Call Local Commit Only
            self.commit_local(f"Update: {self.current_title}")
                
            return True
        except Exception as e:
            print(f"[Error] Save failed: {e}")
            return False

    def update_messages(self, scraped_messages):
        """
        updates self.history to match the order of scraped_messages exactly,
        while strictly preserving the order appearing in HTML.
        
        Logic:
        1. Identify the 'anchor' point where scraped_messages starts in self.history.
        2. If found, splice the history: keep everything before the anchor, and append scraped_messages.
        3. If not found (completely new context or scrolled too far), append or replace based on heuristics.
        """
        if not scraped_messages:
            return

        # 1. Normalize for comparison
        def get_stable_key(msg):
            # precise enough to distinguish messages but loose enough to handle minor scrap diffs
            return f"{msg['role']}:{msg['id']}"

        scraped_keys = [get_stable_key(m) for m in scraped_messages]
        
        # 2. Try to find the start of scraped_messages in existing history
        # We look for the first message of scraped_messages in self.history
        start_msg_key = scraped_keys[0]
        match_index = -1
        
        # Search backwards to find the most recent occurrence (handling loops/re-generations better)
        # But logically, we should search from the end? Or just search all.
        # Let's search all.
        for i, msg in enumerate(self.history):
            if get_stable_key(msg) == start_msg_key:
                match_index = i
                break # Found the first match point
        
        changes_made = False
        
        if match_index != -1:
            # Case A: Overlap Found
            # We trust the scraper's view as the "current truth" for the tail of the conversation.
            # However, we must be careful not to delete history just because we scrolled up.
            
            # Check if scraped_messages significantly deviates from existing history at that point
            # splice_candidate = self.history[:match_index] + scraped_messages
            
            # Sub-case: Is scraped_messages just a subset of existing history? (Scrolling up)
            # Compare scraped_messages with self.history[match_index : match_index + len(scraped)]
            existing_segment = self.history[match_index : match_index + len(scraped_messages)]
            
            is_subset = True
            if len(existing_segment) != len(scraped_messages):
                is_subset = False
            else:
                for old, new in zip(existing_segment, scraped_messages):
                    if get_stable_key(old) != get_stable_key(new):
                        is_subset = False
                        break
            
            if is_subset:
                # We are just looking at a past segment. 
                # Check for content updates (e.g. streaming finished)
                for i, new_msg in enumerate(scraped_messages):
                    hist_idx = match_index + i
                    old_msg = self.history[hist_idx]
                    if len(new_msg['text']) > len(old_msg['text']) or new_msg['text'] != old_msg['text']:
                        self.history[hist_idx] = new_msg
                        changes_made = True
            else:
                # Divergence detected! 
                # The user might have regenerated a response, or edited a prompt in the middle.
                # In this case, the screen shows the NEW reality.
                # We splice: Keep history BEFORE the match, and Replace the rest with scraped_messages.
                
                # Wait, if we are scrolling up, we might see messages 5-10, but history has 1-20.
                # scraped = [5, ... 10]
                # history = [1, ... 20]
                # match_index = 4 (0-based)
                # existing_segment = [5...10]
                # is_subset = True -> No splicing. just update. 
                
                # What if we edited message 10?
                # scraped = [5, ... 10']
                # history = [1, ... 10, 11...20]
                # match_index = 4
                # existing_segment = [5...10] vs [5...10']
                # is_subset = False (because 10 != 10' content might differ, ids same?)
                # If IDs are same, is_subset MIGHT be true depending on get_stable_key.
                # stable_key uses ID. If ID is hash of content, it changes. 
                # If ID is DOM ID, it might stay same. 
                # Gemini DOM IDs usually stable? 
                
                # IF ID changes on edit:
                # scraped = [5, ... 10_new]
                # match_index of 5 is 4.
                # Loop checks: 5==5, 6==6, ... 10!=10_new.
                # is_subset = False.
                
                # Splicing Logic:
                # We trust valid overlap.
                # If we are strictly following HTML order, we should cut off the old branch.
                new_history = self.history[:match_index] + scraped_messages
                
                # Heuristic: Don't lose future history if we just didn't scrape it all?
                # If scraped_messages ends at 10_new, but history had 11..20. 
                # Does the screen show 11..20? The scraper only sees what's visible.
                # If we cut now, we lose 11..20. 
                # This is risky. 
                
                # Refined Logic:
                # Only cut if we have strong evidence of divergence (ID mismatch).
                # If IDs match but text differs, just update text.
                # If ID mismatch implies new branch.
                
                # Let's try to "merge" the tail if possible?
                # No, user wants "HTML Text Order". HTML is truth.
                # If HTML shows A -> B -> C', and History has A -> B -> C -> D.
                # And scraper sees A -> B -> C'.
                # We should probably result in A -> B -> C'. 
                # Because C' implies C was edited/regenerated, invalidating D.
                
                # But what if scraper just sees A -> B (scrolled top) and C -> D is offscreen bottom?
                # Then scraped=[A, B]. match=A. is_subset=True. We do NOT cut.
                
                # Conclusion:
                # If it's a subset (IDs match), we update content and DO NOT cut.
                # If it's divergent (IDs mismatch at some point), we must assume a branch change.
                # BUT, we only see a window. 
                # If scraped=[A, B_new], and history=[A, B_old, C, D].
                # match=A. A==A. B_new != B_old.
                # We splice -> [A, B_new]. We lose C, D. 
                # This is CORRECT for Gemini behavior: Editing B removes C and D.
                
                print(f"[Sync] Divergence/Branch detected at index {match_index}. Splicing.")
                self.history = self.history[:match_index] + scraped_messages
                self.message_map = {msg['id']: i for i, msg in enumerate(self.history)}
                changes_made = True
                
        else:
            # Case B: No Overlap (Completely new or Gap)
            # If history is empty, easy.
            if not self.history:
                self.history = scraped_messages
                changes_made = True
            else:
                # Check if it fits at the end?
                # last_msg = self.history[-1]
                # If completely disjoint, we might be scrolling down after a gap?
                # Or scrolling up past memory?
                # For safety, we append. merging is hard without overlap.
                
                # Try soft match? (Text based)
                # ... skipping for now to simple append.
                # Check if we should append or prepend (rare).
                # Assume append.
                
                # Check for duplicates that got missed by ID match?
                # Trust the ID.
                self.history.extend(scraped_messages)
                changes_made = True
                print(f"[Sync] appended {len(scraped_messages)} messages (No overlap found)")

        # Re-build map and save if changed
        if changes_made:
            self.message_map = {msg['id']: i for i, msg in enumerate(self.history)}
            self.save()

async def main():
    print("==========================================")
    print("      GemKeeper v4.3 - Throttled Push     ")
    print("==========================================")
    
    # Pre-flight check: Cleanup locks
    cleanup_stale_git_locks(KB_ROOT)
    
    async with async_playwright() as p:
        try:
            browser = await p.chromium.connect_over_cdp(CDP_URL)
            print(f"[Conn] Connected to Chrome at {CDP_URL}")
        except Exception as e:
            print(f"[Error] Connection failed: {e}")
            print(f"Make sure Chrome is running with remote-debugging-port=9222")
            return

        context = browser.contexts[0]
        page = None
        for pg in context.pages:
            if TARGET_DOMAIN in pg.url:
                page = pg
                break
        
        if not page:
            print("[Init] Opening new Gemini tab...")
            page = await context.new_page()
            await page.goto(f"https://{TARGET_DOMAIN}/")
        else:
            print(f"[Init] Found Gemini tab")

        active_session = None
        current_chat_id = None
        empty_scrape_streak = 0
        
        # Throttling State
        LAST_PUSH_TIME = datetime.min
        PUSH_INTERVAL_SEC = CONFIG['system'].get('git_push_interval_sec', 300)
        
        def handle_timed_push(session):
            nonlocal LAST_PUSH_TIME
            now = datetime.now()
            if (now - LAST_PUSH_TIME).total_seconds() > PUSH_INTERVAL_SEC:
                print(f"[Git] Checking pending commits for push (Interval: {PUSH_INTERVAL_SEC}s)...")
                if session.push_remote():
                   LAST_PUSH_TIME = now
                else:
                   # If push failed or no remote configured, update time anyway to avoid spamming logic triggers
                   # But strictly speaking we only update if successful or skipping. 
                   # For safety, let's update it to "now" so we retry in X seconds not immediately.
                   LAST_PUSH_TIME = now

        while True:
            try:
                await asyncio.sleep(POLL_INTERVAL)
                
                if page.is_closed():
                    print("[Error] Page closed. Exiting.")
                    break
                
                curr_url = page.url
                
                if TARGET_DOMAIN not in curr_url:
                    continue

                extracted_id = extract_chat_id(curr_url)
                
                # Logic Update: Do not create session if ID is missing (e.g. New Chat page)
                if extracted_id != current_chat_id:
                    print(f"\n[Switch] Detected Change: {current_chat_id} -> {extracted_id}")
                    current_chat_id = extracted_id
                    
                    if extracted_id:
                        active_session = GemSessionV4(extracted_id)
                    else:
                        active_session = None
                
                # Update call: Pass current_chat_id
                messages, candidate_title, header_title = await scrape_page(
                    page, 
                    smart_title_fallback=True, 
                    current_chat_id=current_chat_id
                )
                
                # --- Watchdog Logic ---
                if active_session and current_chat_id != "New_Session":
                    if not messages:
                        empty_scrape_streak += 1
                        if empty_scrape_streak > DOM_Watchdog_Threshold:
                            print("\n" + "="*50)
                            print(" [CRITICAL] DOM Watchdog Triggered!")
                            print(" No messages found for multiple cycles.")
                            print(" DOM structure might have changed.")
                            print(" STOPPING Collection. Please check config.yaml selectors.")
                            print("="*50 + "\n")
                            break
                    else:
                        empty_scrape_streak = 0
                
                if active_session:
                    # Title Priority Logic
                    final_title = None
                    
                    # Priority 1: Header Title
                    if header_title:
                        final_title = header_title
                    
                    # Priority 2: Page Title (if not generic)
                    if not final_title:
                        curr_title = await page.title()
                        is_generic = any(x in curr_title for x in ["Gemini", "Google Gemini", "새 대화", "New Chat"])
                        
                        if not is_generic:
                            final_title = curr_title
                    
                    # Priority 3: Candidate Title (First user message)
                    if not final_title and candidate_title:
                        final_title = candidate_title
                        
                    # Fallback
                    if not final_title:
                        final_title = await page.title()

                    active_session.set_title(final_title)
                    active_session.update_messages(messages)
                    
                    # --- Throttled Push Check ---
                    handle_timed_push(active_session)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[Loop Error] {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    print("[Exit] GemKeeper stopped.")

if __name__ == "__main__":
    asyncio.run(main())
