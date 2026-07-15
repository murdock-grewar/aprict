--------------------------------------
--- Custom commands defined by me: ---
--- Outside of insert mode:
--- <localleader>ee   -- toggles between the way diagnostics are displayed
--- <localleader>ih   -- toggle the display of inlay hints
--- <localleader>ca   -- open a code action menu
--- <localleader>qf   -- do a quickfix
---
--- When in insert mode:
--- <C-space>  -- will open up an autocomplete menu
--------------------------------------

--vim.cmd("highlight Normal guibg=NONE") -- remove errant default of black background that neovim adds over vim
vim.o.smarttab = true
vim.o.softtabstop = 4
vim.o.shiftwidth = 4
vim.o.expandtab = true

vim.wo.number = true -- show line numbers (use wo.relativenumber = true for relative numbers)
vim.opt.fixeol = true -- stop neovim from 'fixing' line endings. It will change the line endings, despite pretending that no modifications were made to a file. The line endings will NOT be fixed if either this is false or if 'binary' is true. Type ":help eol" to see more.
vim.opt.list = true -- show listchars. These are invisible characters like spaces, EOF, etc
vim.opt.binary = false -- dunno. Don't use. See ':help binary' for info.
vim.opt.listchars = {
  tab = "▸·",
  trail = "·",
  nbsp = "~",
  eol = "¬", -- a longstanding bug causes neovim to show an EOL character even when the line/file does not end with one! See https://github.com/vim/vim/issues/6119. Can be seen by doing `printf 'x = 10' > test.py` and opening `test.py` with neovim. An EOL character is added to the end, even though it doesn't really exist in the file.
  lead = "·", -- spaces at the beginning of the line
  space = "·", -- other spaces
}
vim.opt.statusline = "%<%f %{&eol?'':'[noeol]'} %h%m%r%=%k[%{(&fenc==''?&enc:&fenc)}%{(&bomb?',BOM':'')}][U+%04B] %-14.(%l,%c%V%) %P" -- adding a nice status line found online.

vim.opt.signcolumn = "yes:1" -- ensure that the diagnostics column is always present, and has a fixedwidth. This helps with it constantly expanding and shrinking when exiting/entering insert mode.

----Set up autocomplete, using the language server plugins installed----
vim.opt.completeopt = { "menuone", "noselect", "popup" } -- prevent the built-in vim.lsp.completion autotrigger from selecting the first item
-- Create a function to call the autocomplete menu, using "omnicompletion", whatever that is
trigger_omnicompletion = function()
  -- Simulate pressing <C-x><C-o> in insert mode
  local omnifunc = vim.api.nvim_get_option_info2("omnifunc", {})
    if omnifunc.was_set then
      vim.api.nvim_feedkeys(vim.api.nvim_replace_termcodes("<C-x><C-o>", true, false, true), "i", false)
  end
end
-- Type Ctrl-Space to fetch an autocomplete menu.
vim.keymap.set("i", "<C-space>", trigger_omnicompletion, { desc = "trigger autocompletion" })
------------------------------------------------------------------------


---- Change diagnostic symbols displayed by neovim ----
-- for some info, see https://github.com/neovim/neovim/discussions/33573
local signs = { Error = " ", Warn = " ", Hint = " ", Info = " " } -- requires Nerd Fonts. (e.g. Iosevka NF)
local diagnostic_cfg = {
    -- Uncomment one of these three. This will select the display style of the diagnostics messages.
    --virtual_text = true,
    virtual_lines = false,
    open_float = true,

    float = {source = true},

    signs = {
            enable = false,
            text = {
                    ["ERROR"] = signs.Error,
                    ["WARN"] = signs.Warn,
                    ["HINT"] = signs.Hint,
                    ["INFO"] = signs.Info,
            },
            texthl = {
              [vim.diagnostic.severity.ERROR] = 'ErrorMsg',
            },
            linehl = {
              --[vim.diagnostic.severity.ERROR] = 'ErrorMsg',
            },
            numhl = {
              [vim.diagnostic.severity.WARN] = 'WarningMsg',
              [vim.diagnostic.severity.ERROR] = 'ErrorMsg',
            },
            severity_sort = true,
    },
    severity_sort = true,
}
vim.diagnostic.config(diagnostic_cfg)

vim.opt.updatetime = 1000 -- time, in milliseconds, required before CursorHold and CursorHoldI events trigger.
local group = vim.api.nvim_create_augroup('OoO', {})
local function au(typ, pattern, cmdOrFn)
	if type(cmdOrFn) == 'function' then
		vim.api.nvim_create_autocmd(typ, { pattern = pattern, callback = cmdOrFn, group = group })
	else
		vim.api.nvim_create_autocmd(typ, { pattern = pattern, command = cmdOrFn, group = group })
	end
end

vim.keymap.set({ "n", "x" }, "<leader>ee", function()
        diagnostic_cfg.open_float = not diagnostic_cfg.open_float
        diagnostic_cfg.virtual_lines = not diagnostic_cfg.virtual_lines
	vim.diagnostic.config(diagnostic_cfg)
end, { noremap = true, silent = false })
au({ 'CursorHold', 'InsertLeave' }, nil, function()
        local opts = {
                focusable = false,
                --scope = 'cursor',
                scope = 'line',
                close_events = { 'BufLeave', 'CursorMoved', 'InsertEnter' },
        }
        if diagnostic_cfg.open_float then
            vim.diagnostic.open_float(nil, opts)
        end
end)

---- Customise the LSP autocompletion ----
vim.lsp.inlay_hint.enable(false)
vim.keymap.set({ "n", "x" }, "<leader>ih", function()
    vim.lsp.inlay_hint.enable(not vim.lsp.inlay_hint.is_enabled())
end, { noremap = true, silent = false })
--vim.lsp.inlay_hint.enable(false)
--vim.lsp.completion.enable(true)

---- Work around an annoying bug that makes the diagnostic column disappear (and shift the text around!) when entering insert mode----
--See this GitHub issue (wontfix): https://github.com/neovim/neovim/issues/26078\
vim.diagnostic.config{
  update_in_insert = false,--true,
}

---- Enable a quickfix shortcut ----
local function quickfix()
    vim.lsp.buf.code_action({
        filter = function(a) return a.isPreferred end,
        apply = true
    })
end
vim.keymap.set('n', '<leader>qf', quickfix, opts)

-- Open a code action menu with <leader>ca
vim.keymap.set({ "n", "x" }, "<leader>ca", function()
    require("tiny-code-action").code_action()
end, { noremap = true, silent = true })


-- Add ruff python linter
vim.lsp.config('ruff', {
  init_options = {
    settings = {
      configurationPreference = "filesystemFirst", -- ensure that local project settings inside a ruff.toml ar pyproject.toml will take precedence over these global settings
      -- Ruff language server settings go here
      showSyntaxErrors = false, -- turning syntax errors off because these are also detected by ty, which results in duplicate errors
      logLevel = 'debug', -- Have this default setting so that there's at least one setting, so that I don't get a warning when I open a .py file because of 'invalid client settings'
    }
  },
  --on_attach = on_attach_fun
})
vim.lsp.enable('ruff')

-- Add ty python type checker
vim.lsp.config('ty', {
  settings = {
    ty = {
      -- ty language server settings go here
    }
  }
})
vim.lsp.enable('ty')


---- Set up the lazy.nvim plugin manager ----
-- Bootstrap lazy.nvim
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not (vim.uv or vim.loop).fs_stat(lazypath) then
  local lazyrepo = "https://github.com/folke/lazy.nvim.git"
  local out = vim.fn.system({ "git", "clone", "--filter=blob:none", "--branch=stable", lazyrepo, lazypath })
  if vim.v.shell_error ~= 0 then
    vim.api.nvim_echo({
      { "Failed to clone lazy.nvim:\n", "ErrorMsg" },
      { out, "WarningMsg" },
      { "\nPress any key to exit..." },
    }, true, {})
    vim.fn.getchar()
    os.exit(1)
  end
end
vim.opt.rtp:prepend(lazypath)

-- Make sure to setup `mapleader` and `maplocalleader` before
-- loading lazy.nvim so that mappings are correct.
vim.g.mapleader = " "
vim.g.maplocalleader = "\\"

-- Setup lazy.nvim
require("lazy").setup({
    spec = {
        {import = "plugins"},
    },
    lockfile = vim.fn.stdpath("config") .. "/lazy-lock.json",
    -- Configure any other settings here. See the documentation for more details.
    -- colorscheme that will be used when installing plugins.
    install = { colorscheme = { "habamax" } },
    -- automatically check for plugin updates
    checker = { enabled = false },
})

-- Set a whole colour scheme --
vim.cmd("colorscheme kanagawa")
vim.cmd("highlight Normal guibg=NONE") -- remove errant default of black background that neovim adds over vim. Restores nice transparency in macOS terminal window.
