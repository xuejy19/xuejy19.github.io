var searchFunc = function (path, search_id, content_id) {
    'use strict';
    var input = document.getElementById(search_id);
    var result = document.getElementById(content_id);
    if (!input || !result) return;
    var entries = [];
    var ready = false;
    var failed = false;
    function render() {
        result.textContent = '';
        var query = input.value.trim().toLowerCase();
        if (!query) return;
        if (!ready) {
            result.textContent = failed ? '搜索暂时无法加载，请刷新后重试。' : '正在搜索…';
            return;
        }
        var keywords = query.split(/\s+/);
        var list = document.createElement('ul');
        list.className = 'search-result-list';
        entries.forEach(function (entry) {
            var haystack = (entry.title + ' ' + entry.content).toLowerCase();
            if (!keywords.every(function (word) { return haystack.indexOf(word) >= 0; })) return;
            var item = document.createElement('li');
            var link = document.createElement('a');
            link.className = 'search-result-title';
            link.href = entry.url;
            link.textContent = entry.title;
            item.appendChild(link);
            var snippet = document.createElement('p');
            snippet.className = 'search-result';
            var index = entry.content.toLowerCase().indexOf(keywords[0]);
            var start = Math.max(0, index - 30);
            snippet.textContent = entry.content.slice(start, start + 130) + '…';
            item.appendChild(snippet);
            list.appendChild(item);
        });
        if (!list.childNodes.length) result.textContent = '没有找到相关内容。';
        else result.appendChild(list);
    }
    // Register before the request so a fast first query survives slow index loading.
    input.addEventListener('input', render);
    $.ajax({
        url: path,
        dataType: 'xml',
        success: function (xml) {
            entries = $('entry', xml).map(function () {
                return { title: $('title', this).text(), content: $('content', this).text(), url: $('url', this).text() };
            }).get();
            ready = true;
            render();
        },
        error: function () { failed = true; render(); }
    });
};
