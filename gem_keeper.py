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
        def normalize(text):
            return ' '.join(text.split())

        changes_made = False
        new_count = 0
        
        for new_msg in scraped_messages:
            msg_id = new_msg['id']
            
            # Case A: ID Match (Exact same ID found)
            if msg_id in self.message_map:
                idx = self.message_map[msg_id]
                existing_msg = self.history[idx]
                
                # Update if new content is longer (Streaming progress)
                if len(new_msg['text']) > len(existing_msg['text']):
                    self.history[idx] = new_msg
                    changes_made = True
            
            else:
                # Case B: ID Mismatch but Potential Continuation
                is_merged = False
                
                if self.history:
                    last_idx = len(self.history) - 1
                    last_msg = self.history[last_idx]
                    
                    # Condition 1: Same Role
                    if last_msg['role'] == new_msg['role']:
                        should_merge = False
                        
                        if new_msg['role'] == 'model':
                            # Aggressive Merge for Model
                            should_merge = True
                        else:
                            # User: Check for text similarity
                            norm_old = normalize(last_msg['text'])
                            norm_new = normalize(new_msg['text'])
                            if norm_new.startswith(norm_old) or norm_old.startswith(norm_new):
                                should_merge = True
                                
                        if should_merge:
                            # Strategy: Delete Old & Append New
                            if last_msg['id'] in self.message_map:
                                del self.message_map[last_msg['id']]
                            
                            self.history.pop()
                            self.history.append(new_msg)
                            self.message_map[msg_id] = len(self.history) - 1
                            
                            changes_made = True
                            is_merged = True

                # Case C: Genuine New Message
                if not is_merged:
                    self.history.append(new_msg)
                    self.message_map[msg_id] = len(self.history) - 1
                    changes_made = True
                    new_count += 1
                    role_icon = "👤" if new_msg['role'] == 'user' else "🤖"
                    print(f"[New] {role_icon} Added: {new_msg['text'][:40]}...")

        if changes_made:
            self.save()
            if new_count > 0:
                print(f"[Sync] Saved {len(self.history)} messages. (+{new_count} new)")

async def main():
    print("==========================================")
    print("      GemKeeper v4.3 - Throttled Push     ")
    print("==========================================")
    
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
                target_session_id = extracted_id if extracted_id else "New_Session"
                
                if target_session_id != current_chat_id:
                    print(f"\n[Switch] Detected Change: {current_chat_id} -> {target_session_id}")
                    current_chat_id = target_session_id
                    active_session = GemSessionV4(target_session_id)
                
                # Update call: Pass current_chat_id
                messages, candidate_title, header_title = await scrape_page(
                    page, 
                    smart_title_fallback=True, 
                    current_chat_id=target_session_id
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
