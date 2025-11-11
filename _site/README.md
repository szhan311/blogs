# 极简 Jekyll 模板（适配 GitHub Pages）

## 使用
1. 将本项目所有文件上传到你的 GitHub 仓库：`你的用户名.github.io`（公开）。
2. 进入仓库 Settings → Pages，确认构建来源（Deploy from a branch 或 GitHub Actions）。
3. 片刻后访问 `https://你的用户名.github.io`。

> 如果用于**项目页**（仓库名不是 `username.github.io`）：
> - 打开 `_config.yml`，设置 `baseurl` 与 `url`；
> - 最终访问地址为 `https://你的用户名.github.io/你的仓库名/`。

## 本地预览（可选）
需要 Ruby 环境：
```bash
gem install bundler jekyll
bundle install
bundle exec jekyll serve
```
