"""Audited terminal-only control flow and bounded native graph-schema retries."""
import ast,hashlib,json,time

def terminal_tree(source):
    tree=ast.parse(source)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    matches=[n for n in ast.walk(main) if isinstance(n,ast.If)
        and ast.unparse(n.test)=='len(episode_success) == args.num_episodes']
    assert len(matches)==1 and ast.unparse(matches[0].body[0])=='finished = True'
    assert len(matches[0].body)==1,'Native terminal branch changed'
    matches[0].body.append(ast.copy_location(ast.Break(),matches[0].body[0]))
    return ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[]))

def install_terminal(entry,path):
    source=path.read_text();tree=terminal_tree(source)
    exec(compile(tree,str(path),'exec'),entry.__dict__)
    return {'native_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'adapted_AST_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
        'change':'One break after finished=True in final completed-episode branch; all navigation-path statements unchanged'}

def install_schema(module,path,out):
    source=path.read_text();tree=ast.parse(source)
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='goal_decomposition')
    handlers=[n for n in ast.walk(method) if isinstance(n,ast.ExceptHandler)]
    assert len(handlers)==1 and ast.unparse(handlers[0].type)=='json.JSONDecodeError'
    handler=handlers[0]
    assert ast.unparse(handler.body[0])=='attempts += 1'
    handler.type=ast.parse('(json.JSONDecodeError, TypeError, KeyError, AttributeError)',mode='eval').body
    handler.name='_schema_error'
    handler.body.insert(0,ast.parse('_benchmark_schema_event(_schema_error)').body[0])
    def event(error):
        with (out/'model_schema_events.jsonl').open('a') as stream:
            stream.write(json.dumps({'time':time.time(),'stage':'native_goalgraph_decomposition',
                'error':type(error).__name__,'handling':'Same original prompt, at most10 native attempts; original fallback retained'})+'\n')
    namespace=dict(module.__dict__);namespace['_benchmark_schema_event']=event
    modified=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    exec(compile(modified,str(path),'exec'),namespace)
    module.GoalGraphDecomposer.goal_decomposition=namespace['goal_decomposition']
    return {'native_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'adapted_AST_sha256':hashlib.sha256(ast.dump(modified,include_attributes=False).encode()).hexdigest(),
        'change':'Invalid native JSON graph shapes join the existing10-attempt retry/fallback path; prompt, valid-response processing and navigation policy unchanged'}
